# standard
import json
import logging
import uuid
from pathlib import Path
from typing import Any

# extern
import bm25s
import chromadb
from pydantic import ValidationError, validate_call
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# local
from ..defines import (
    BM25_DIRECTORY,
    CHROMA_DIRECTORY,
    CHUNKS_METADATA_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_SAVE_DIRECTORY,
)
from ..utils.logger import LoggerManager
from ..services.translator import Translator
from ..schemas.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

MAX_CONTENT_LENGTH: int = 200

BM25_SCORE_WEIGHT: float = 0.65
CHROMA_SCORE_WEIGHT: float = 0.35

RRF_K: int = 60
CANDIDATE_MULTIPLIER: int = 42


class Searcher:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        lm: LoggerManager,
        console: Console,
        embedding_model: Any,
        translator: Translator,
        query: str = "",
        dataset_path: str = DEFAULT_DATASET_PATH,
        save_directory: str = DEFAULT_SAVE_DIRECTORY,
        k: int = 5,
        question_id: str = "",
    ) -> None:
        self.lm = lm
        self.console = console

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if not BM25_DIRECTORY.exists():
            raise FileNotFoundError("BM25 index directory does not exist")
        if not BM25_DIRECTORY.is_dir():
            raise NotADirectoryError("BM25 path is not a directory")

        if not CHUNKS_METADATA_PATH.exists():
            raise FileNotFoundError("chunks metadata file does not exist")
        if not CHUNKS_METADATA_PATH.is_file():
            raise FileNotFoundError("chunks metadata path is not a file")

        try:
            with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                self.chunks_metadata: dict[str, dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid metadata JSON file: {CHUNKS_METADATA_PATH}"
            ) from e

        self.embedding_model = embedding_model
        self.retriever = bm25s.BM25.load(
            str(BM25_DIRECTORY),
            load_corpus=True,
        )

        corpus_size = len(
            self.retriever.corpus
        ) if self.retriever.corpus else 0
        if corpus_size == 0:
            raise ValueError("BM25 corpus is empty")

        self.k = min(k, corpus_size)
        self.candidate_k = min(k * CANDIDATE_MULTIPLIER, corpus_size)

        self.query = query
        self.translator = translator
        self.translated_query: str = ""
        self.question_id = question_id

        self.dataset_path = dataset_path
        self.save_directory = save_directory

    def _bm25_ids(self, show_progress: bool = False) -> list[str]:
        query_tokens = bm25s.tokenize(
            self.translated_query, show_progress=show_progress,
        )

        results, _ = self.retriever.retrieve(
            query_tokens, k=self.candidate_k, show_progress=show_progress,
        )

        return [result["id"] for result in results[0]]

    def _chroma_ids(self) -> list[str]:
        client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
        collection = client.get_collection(name="chunks")

        query_embedding = self.embedding_model.encode(
            self.translated_query, convert_to_numpy=True,
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.candidate_k,
        )

        ids = results.get("ids", [[]])
        return ids[0]

    @staticmethod
    def _rrf(rankings: list[tuple[list[str], float]]) -> list[str]:
        scores: dict[str, float] = {}

        for ranking, weight in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                    weight / (RRF_K + rank)
                )

        return sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

    def search(
        self,
        show_progress: bool = False,
        show_bm25_progress: bool = False
    ) -> MinimalSearchResults:
        self.lm.logger.debug("Searching %r with k=%d", self.query, self.k)

        ids: list[tuple[list[str], float]] = []
        show_rich_progress = show_progress and self._should_show_progress()

        with Progress(
            SpinnerColumn("shark", style="cyan"),
            TextColumn("[black]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
            disable=not show_rich_progress,
        ) as progress:
            task_id = progress.add_task(
                "[black]Searching [cyan]0/5",
                total=5,
            )

            progress.update(
                task_id,
                description="[black]Searching [cyan]1/5 translating query",
            )
            self.translated_query = self.translator.translate_to_english(
                self.query,
            )
            self.lm.logger.debug(
                "Translated query: %s", self.translated_query
            )
            progress.advance(task_id)

            progress.update(
                task_id,
                description="[black]Searching [cyan]2/5 BM25",
            )
            ids.append((
                self._bm25_ids(
                    show_progress=(
                        show_bm25_progress and not show_rich_progress
                    )
                ),
                BM25_SCORE_WEIGHT,
            ))
            progress.advance(task_id)

            progress.update(
                task_id,
                description="[black]Searching [cyan]3/5 Chroma",
            )
            if CHROMA_DIRECTORY.exists():
                if CHROMA_DIRECTORY.is_dir():
                    try:
                        ids.append((self._chroma_ids(), CHROMA_SCORE_WEIGHT))
                    except Exception as e:
                        self.lm.logger.warning("Chroma search failed: %s", e)
                else:
                    self.lm.logger.warning(
                        "Chroma path exists but is not a directory: %s",
                        CHROMA_DIRECTORY,
                    )
            progress.advance(task_id)

            progress.update(
                task_id,
                description="[black]Searching [cyan]4/5 merging results",
            )
            merged_ids = self._rrf(ids)
            selected_ids = merged_ids[:self.k]

            self.lm.logger.debug("Selected ids: %s", selected_ids)
            progress.advance(task_id)

            progress.update(
                task_id,
                description="[black]Searching [cyan]5/5 building sources",
            )
            sources: list[MinimalSource] = []

            for chunk_id in selected_ids:
                md = self.chunks_metadata.get(chunk_id)
                if md is None:
                    self.lm.logger.warning(
                        "Missing md for chunk id %s", chunk_id
                    )
                    continue

                try:
                    sources.append(
                        MinimalSource(
                            file_path=md["file_path"],
                            first_character_index=md[
                                "first_character_index"
                            ],
                            last_character_index=md["last_character_index"],
                        )
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Invalid metadata for chunk id {chunk_id}: "
                        f"missing {e}"
                    ) from e

            progress.advance(task_id)
            progress.update(
                task_id,
                description="[green]Searched [green]5/5",
            )

        return MinimalSearchResults(
            question_id=self.question_id or str(uuid.uuid4()),
            question=self.translated_query,
            retrieved_sources=sources,
        )

    def _should_show_progress(self) -> bool:
        return (
            self.console.is_terminal
            and not self.lm.logger.isEnabledFor(logging.INFO)
        )

    def search_dataset(self) -> None:
        self.lm.logger.debug(
            "Searching in %s with k=%d", self.dataset_path, self.k,
        )

        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: {dataset_path}"
            )
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset path is not a file: {dataset_path}"
            )

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_dataset = json.load(f)

            dataset = RagDataset.model_validate(raw_dataset)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON file: {dataset_path}"
            ) from e

        except ValidationError as e:
            raise ValueError(
                f"Invalid dataset format: {dataset_path}"
            ) from e

        results: list[MinimalSearchResults] = []
        total = len(dataset.rag_questions)
        show_progress = self._should_show_progress()

        with Progress(
            SpinnerColumn("shark", style="cyan"),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=not show_progress,
        ) as progress:
            task_id = progress.add_task(
                f"[black]Searching [cyan]0/{total}", total=total
            )

            for index, question in enumerate(dataset.rag_questions, start=1):
                progress.update(
                    task_id, description=(
                        f"[black]Searching [cyan]{index}/{total}"
                    )
                )

                self.query = question.question
                self.question_id = question.question_id

                result = self.search(show_bm25_progress=False)
                results.append(result)

                progress.advance(task_id)

            progress.update(
                task_id, description=f"[green]Searched [green]{total}/{total}"
            )

        student_results = StudentSearchResults(
            search_results=results, k=self.k
        )

        save_directory = Path(self.save_directory)
        if save_directory.exists() and not save_directory.is_dir():
            raise NotADirectoryError(
                f"Save directory path is not a directory: {save_directory}"
            )

        save_directory.mkdir(parents=True, exist_ok=True)

        output_path = save_directory / dataset_path.name

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(student_results.model_dump_json(indent=4))

        self.lm.logger.info("Saved search results to %s", output_path)
