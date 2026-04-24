# standard imports
import json
from typing import Any

# extern imports
import bm25s
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import validate_call
from transformers import pipeline

# local imports
from .logger import LoggerManager
from .defines import (
    BM25_DIRECTORY, CHROMA_DIRECTORY, CHUNKS_METADATA_PATH,
    EMBEDDING_MODEL, TRANSLATION_MODEL
)
from .types import MinimalSearchResults, MinimalSource

MAX_CONTENT_LENGTH: int = 200
BM25_SCORE_WEIGHT: float = 0.75
CHROMA_SCORE_WEIGHT: float = 0.25
RRF_K: int = 60


class Translator:
    def __init__(self) -> None:
        self.translator = pipeline(task="translation", model=TRANSLATION_MODEL)

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.split())

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        normalized: str = self._normalize(text)
        result = self.translator(normalized, max_length=512)
        return str(result[0]["translation_text"])


class Searcher():
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, query: str, lm: LoggerManager, k: int = 5
    ) -> None:
        self.lm = lm

        if not BM25_DIRECTORY.exists():
            raise FileNotFoundError(
                "TODO"
            )
        if not BM25_DIRECTORY.is_dir():
            raise FileNotFoundError(
                "TODO"
            )
        # maybe cannot access case, maybe do a helper func
        # eventually use path.resolve()
        self.retriever = bm25s.BM25.load(str(BM25_DIRECTORY), load_corpus=True)

        self.k = min(
            k, len(self.retriever.corpus) if self.retriever.corpus else 0
        )

        self.query = query
        self.translated_query: str = ""

    def _bm25_ids(self) -> list[str]:
        query_tokens = bm25s.tokenize(self.translated_query)
        results, _ = self.retriever.retrieve(query_tokens, k=self.k)

        return [result["id"] for result in results[0]]

    def _chroma_ids(self) -> list[str]:
        client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
        collection = client.get_collection(name="chunks")

        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        query_embedding = embedding_model.encode(
            self.translated_query, convert_to_numpy=True,
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=self.k
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
            scores, key=lambda chunk_id: scores[chunk_id], reverse=True
        )

    # change output
    def _format_result(
        self, search_result: MinimalSearchResults,
        chunks_metadata: dict[str, dict[str, Any]],
        selected_ids: list[str],
    ) -> str:
        lines: list[str] = []

        lines.append("=" * 80)
        lines.append("SEARCH RESULT")
        lines.append("=" * 80)
        lines.append(f"question_id: {search_result.question_id}")
        lines.append(f"question:    {search_result.question}")
        lines.append(f"k:           {len(search_result.retrieved_sources)}")
        lines.append("")

        for index, (source, chunk_id) in enumerate(
            zip(search_result.retrieved_sources, selected_ids),
            start=1,
        ):
            metadata = chunks_metadata[chunk_id]
            content = str(metadata.get("content", ""))

            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n[...]"

            source_json = source.model_dump_json(indent=4)

            lines.append("-" * 80)
            lines.append(f"RESULT #{index}")
            lines.append("-" * 80)
            lines.append("source:")
            lines.append(source_json)
            lines.append("")
            lines.append("content:")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def search(self) -> MinimalSearchResults:
        self.lm.logger.debug("Searching %r with k=%d", self.query, self.k)

        translator = Translator()
        self.translated_query = translator.translate_to_english(self.query)
        self.lm.logger.debug("Translated query: %s", self.translated_query)

        ids: list[tuple[list[str], float]] = []

        ids.append((self._bm25_ids(), BM25_SCORE_WEIGHT))
        if CHROMA_DIRECTORY.exists():
            ids.append((self._chroma_ids(), CHROMA_SCORE_WEIGHT))

        merged_ids = self._rrf(ids)
        selected_ids = merged_ids[:self.k]

        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            chunks_metadata: dict[str, dict[str, Any]] = json.load(f)

        sources: list[MinimalSource] = []

        for chunk_id in selected_ids:
            metadata = chunks_metadata[chunk_id]

            sources.append(
                MinimalSource(
                    file_path=metadata["file_path"],
                    first_character_index=metadata["first_character_index"],
                    last_character_index=metadata["last_character_index"],
                )
            )

        msr = MinimalSearchResults(
            question_id="manual",
            question=self.translated_query,
            retrieved_sources=sources
        )

        print(self._format_result(msr, chunks_metadata, selected_ids))
        return msr
