# standard import
import re
import json
import shutil
import logging
from pathlib import Path
from typing import Any

# extern imports
import bm25s
import chromadb
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from pydantic import (
    validate_call, PositiveInt, BaseModel,
    Field, ValidationError, ConfigDict
)

# local imports
from .text_splitter import TextSplitter
from .logger import LoggerManager
from .hash import md5sum, file_md5sum
from .defines import (
    OUTPUT_DIRECTORY, BM25_DIRECTORY, CHROMA_DIRECTORY, CHUNKS_METADATA_PATH,
    MANIFEST_PATH, MAX_BATCH_SIZE, EMBEDDING_MODEL, DEFAULT_VLLM
)


class ChunkMetadata(BaseModel):
    content: str
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


class CachedFile(BaseModel):
    stores: set[str] = Field(default_factory=set)
    file_path: str
    file_hash: str
    chunks_ids: set[str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(0, ge=0)
    llm_model: str = EMBEDDING_MODEL
    extensions: list[str] = Field(default_factory=list)
    vllm: str = DEFAULT_VLLM
    files_by_extensions: dict[str, dict[str, CachedFile]] = Field(
        default_factory=dict
    )

    @staticmethod
    def existing_manifest_data() -> dict[str, Any]:
        if not MANIFEST_PATH.exists():
            return {}

        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"Manifest path is not a file: {MANIFEST_PATH}"
            )

        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid manifest JSON file: {MANIFEST_PATH}"
            ) from e

        if not isinstance(manifest_data, dict):
            raise ValueError(
                f"Invalid manifest format: expected object in {MANIFEST_PATH}"
            )

        return manifest_data

    def _remove_extensions(self, extensions: set[str]) -> list[str]:
        delete_chunks_ids: list[str] = []

        for ext in list(self.files_by_extensions.keys()):
            if "*" not in extensions and ext not in extensions:
                for cached_file in self.files_by_extensions[ext].values():
                    delete_chunks_ids.extend(cached_file.chunks_ids)

                del self.files_by_extensions[ext]

                if ext in self.extensions:
                    self.extensions.remove(ext)

        return delete_chunks_ids

    def _remove_missing_files(self) -> list[str]:
        delete_chunks_ids: list[str] = []

        for ext in list(self.files_by_extensions.keys()):
            for file_id in list(self.files_by_extensions[ext].keys()):
                cached_file = self.files_by_extensions[ext][file_id]
                path = Path(cached_file.file_path)

                if not path.exists():
                    delete_chunks_ids.extend(cached_file.chunks_ids)
                    del self.files_by_extensions[ext][file_id]

            if not self.files_by_extensions[ext]:
                del self.files_by_extensions[ext]

                if ext in self.extensions:
                    self.extensions.remove(ext)

        return delete_chunks_ids

    @classmethod
    def load(
        cls, chunk_size: int, extensions: set[str]
    ) -> tuple["Manifest", list[str]]:
        if not MANIFEST_PATH.exists():
            return cls(chunk_size=chunk_size, llm_model=EMBEDDING_MODEL), []

        manifest = cls(**cls.existing_manifest_data())
        delete_chunks_ids: list[str] = []

        if (
            manifest.chunk_size != chunk_size
            or manifest.llm_model != EMBEDDING_MODEL
            or manifest.vllm != DEFAULT_VLLM
        ):
            for files_by_id in manifest.files_by_extensions.values():
                for cached_file in files_by_id.values():
                    delete_chunks_ids.extend(cached_file.chunks_ids)

            return (
                cls(chunk_size=chunk_size, llm_model=EMBEDDING_MODEL),
                delete_chunks_ids
            )

        delete_chunks_ids.extend(manifest._remove_extensions(extensions))
        delete_chunks_ids.extend(manifest._remove_missing_files())

        return manifest, delete_chunks_ids

    def add_store(
        self, chunks_metadata: dict[str, dict[str, Any]],
        chunks_ids: list[str], store: str,
    ) -> None:
        for chunk_id in chunks_ids:
            metadata = chunks_metadata[chunk_id]

            file_path = Path(metadata["file_path"])
            file_id: str = md5sum(str(file_path))
            file_suffix: str = file_path.suffix.removeprefix(".").lower()

            manifest_files = self.files_by_extensions.get(file_suffix, {})
            manifest_file = manifest_files.get(file_id)

            if manifest_file is None:
                continue

            manifest_file.chunks_ids.add(chunk_id)
            manifest_file.stores.add(store)

    def sync_files(
        self, files: list[Path]
    ) -> tuple[list[str], set[str], set[str]]:
        delete_chunks_ids: list[str] = []
        updated_files_ids: set[str] = set()
        new_files_ids: set[str] = set()

        for file in files:
            file_id: str = md5sum(str(file))
            file_hash: str = file_md5sum(file)
            file_suffix: str = file.suffix.removeprefix(".").lower()

            manifest_files = self.files_by_extensions.setdefault(
                file_suffix, {}
            )
            manifest_file = manifest_files.get(file_id)

            if manifest_file is None:
                manifest_file = CachedFile(
                    file_path=str(file),
                    file_hash=file_hash,
                    chunks_ids=set(),
                )
                manifest_files[file_id] = manifest_file
                new_files_ids.add(file_id)

            elif manifest_file.file_hash != file_hash:
                delete_chunks_ids.extend(manifest_file.chunks_ids)
                manifest_file.chunks_ids = set()
                manifest_file.stores.clear()
                updated_files_ids.add(file_id)

            manifest_file.file_path = str(file)
            manifest_file.file_hash = file_hash

        return delete_chunks_ids, updated_files_ids, new_files_ids


class Indexer:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, directory_path: str, lm: LoggerManager, console: Console,
        extensions: str = "*", chunk_size: PositiveInt = 2000,
        idiot: bool = False
    ) -> None:
        self.directory_path = Path(directory_path)

        self.lm = lm
        self.console = console
        self.chunk_size = chunk_size
        self.idiot = idiot

        self.extensions = self._parse_extensions(extensions)
        self.delete_chunks_ids: list[str] = []
        self.updated_files_ids: set[str] = set()
        self.new_files_ids: set[str] = set()

        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"Path does not exist: {self.directory_path}"
            )

        if not self.directory_path.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {self.directory_path}"
            )

    @staticmethod
    def _parse_extensions(extensions: str) -> set[str]:
        if extensions.strip() == "*":
            return {"*"}

        parsed_extensions: set[str] = set()

        for extension in extensions.split(":"):
            e: str = re.sub(r"[^a-zA-Z0-9]", "", extension).lower()
            if e:
                parsed_extensions.add(e)

        if not parsed_extensions:
            raise ValueError(
                "extensions must contain at least one valid extension or '*'"
            )

        return parsed_extensions

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []

        for path in self.directory_path.rglob("*"):
            if not path.is_file():
                continue
            extension: str = path.suffix.removeprefix(".").lower()
            if "*" not in self.extensions and extension not in self.extensions:
                continue

            files.append(path)

        return files

    def _load_document(self, path: Path) -> Document:
        content: str = path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            page_content=content,
            metadata={"source": str(path), "suffix": path.suffix},
        )

    def _split_into_chunks(
        self, files: list[Path]
    ) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:

        chunks_content: list[str] = []
        chunks_metadata: dict[str, dict[str, Any]] = {}
        chunks_ids: list[str] = []

        for file in files:
            file_id: str = md5sum(str(file))

            doc: Document = self._load_document(file)
            # self.lm.logger.debug("Loaded '%s' file", str(file))

            splitter: TextSplitter = TextSplitter.from_filename(
                str(file), chunk_size=self.chunk_size,
            )
            file_chunks: list[Document] = splitter.split_documents([doc])

            source_text: str = doc.page_content
            search_from: int = 0

            for chunk in file_chunks:
                content: str = chunk.page_content
                chunks_content.append(content)

                first_character_index: int | None = (
                    chunk.metadata.get("start_index")
                )

                if first_character_index is None:
                    first_character_index = (
                        source_text.find(content, search_from)
                    )
                    if first_character_index == -1:
                        raise ValueError(
                            f"Unable to locate chunk in source file: {file}"
                        )

                last_character_index: int = (
                    first_character_index + len(content)
                )

                chunk_id: str = (
                    "chunk"
                    f"_{file_id}"
                    f"_{first_character_index}"
                    f"_{last_character_index}"
                )

                chunks_metadata[chunk_id] = (
                    ChunkMetadata(
                        content=content,
                        file_path=str(file),
                        first_character_index=first_character_index,
                        last_character_index=last_character_index,
                    ).model_dump()
                )

                chunks_ids.append(chunk_id)
                search_from = first_character_index + 1

        return chunks_content, chunks_metadata, chunks_ids

    def _bm25_index(
        self, chunks_content: list[str], chunks_ids: list[dict[str, str]]
    ) -> None:
        if not chunks_content or not chunks_ids:
            raise ValueError("No chunks to index with BM25")

        if BM25_DIRECTORY.exists():
            if not BM25_DIRECTORY.is_dir():
                raise NotADirectoryError(
                    f"BM25 path is not a directory: {BM25_DIRECTORY}"
                )
            shutil.rmtree(BM25_DIRECTORY)

        BM25_DIRECTORY.mkdir(parents=True, exist_ok=True)

        corpus_tokens = bm25s.tokenize(chunks_content, show_progress=False)
        retriever = bm25s.BM25(corpus=chunks_ids)
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(str(BM25_DIRECTORY))

    def _chroma_filter(
        self, chunks_content: list[str],
        chunks_metadata: dict[str, dict[str, Any]],
        chunks_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        chroma_chunks_content: list[str] = []
        chroma_chunks_ids: list[str] = []

        chroma_store_missing: bool = not CHROMA_DIRECTORY.exists()

        if self.idiot and chroma_store_missing:
            return chroma_chunks_content, chroma_chunks_ids

        for content, chunk_id in zip(chunks_content, chunks_ids):
            metadata = chunks_metadata[chunk_id]

            file_path = Path(metadata["file_path"])
            file_id: str = md5sum(str(file_path))
            file_suffix: str = file_path.suffix.removeprefix(".").lower()

            manifest_files = self.manifest.files_by_extensions.get(
                file_suffix, {}
            )
            manifest_file = manifest_files.get(file_id)

            if manifest_file is None:
                continue

            if self.idiot and file_id not in self.updated_files_ids:
                continue

            if (
                chroma_store_missing
                or file_id in self.updated_files_ids
                or "chroma" not in manifest_file.stores
                or chunk_id not in manifest_file.chunks_ids
            ):
                chroma_chunks_content.append(content)
                chroma_chunks_ids.append(chunk_id)

        return chroma_chunks_content, chroma_chunks_ids

    def _count_updated_chroma_chunks(
        self, chunks_metadata: dict[str, dict[str, Any]],
        chunks_ids: list[str]
    ) -> int:
        count: int = 0

        for chunk_id in chunks_ids:
            metadata = chunks_metadata[chunk_id]

            file_path = Path(metadata["file_path"])
            file_id: str = md5sum(str(file_path))

            if file_id in self.updated_files_ids:
                count += 1

        return count

    def _should_show_progress(self) -> bool:
        return (
            self.console.is_terminal
            and not self.lm.logger.isEnabledFor(logging.INFO)
        )

    def _chroma_index(
        self, chunks_content: list[str], chunks_ids: list[str]
    ) -> None:
        if CHROMA_DIRECTORY.exists() and not CHROMA_DIRECTORY.is_dir():
            raise NotADirectoryError(
                f"Chroma path is not a directory: {CHROMA_DIRECTORY}"
            )

        CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))

        collection = client.get_or_create_collection(name="chunks")

        if self.delete_chunks_ids:
            for i in range(0, len(self.delete_chunks_ids), MAX_BATCH_SIZE):
                batch_delete_ids = self.delete_chunks_ids[i:i + MAX_BATCH_SIZE]
                collection.delete(ids=batch_delete_ids)

        if not chunks_ids:
            return

        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.lm.logger.debug(
            "Embedding model device: %s", embedding_model.device
        )

        total = (len(chunks_content) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        show_progress = self._should_show_progress()

        with Progress(
            SpinnerColumn("shark", style="cyan"),
            TextColumn("[black]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=not show_progress,
        ) as progress:
            task_id = progress.add_task(
                f"[black]Embedding [cyan]0/{total}",
                total=total,
            )

            for batch_index, i in enumerate(
                range(0, len(chunks_content), MAX_BATCH_SIZE),
                start=1,
            ):
                progress.update(
                    task_id,
                    description=(
                        f"[black]Embedding [cyan]{batch_index}/{total}"
                    )
                )

                batch_content = chunks_content[i:i + MAX_BATCH_SIZE]
                batch_ids = chunks_ids[i:i + MAX_BATCH_SIZE]

                batch_embeddings = embedding_model.encode(
                    batch_content,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                collection.add(
                    embeddings=batch_embeddings.tolist(),
                    ids=batch_ids,
                )

                progress.advance(task_id)

            progress.update(
                task_id, description=f"[green]Embedded [green]{total}/{total}"
            )

    def _store_chunks(
        self, chunks_metadata: dict[str, dict[str, Any]]
    ) -> None:
        CHUNKS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks_metadata, f)

    def _store_manifest(self) -> None:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.manifest.model_dump(mode="json"), f, indent=4)

    def index_directory(self) -> None:
        self.lm.logger.debug(
            "Indexing %s with chunk size %d",
            str(self.directory_path), self.chunk_size
        )

        # load or create manifest
        try:
            self.manifest, self.delete_chunks_ids = (
                Manifest.load(self.chunk_size, self.extensions)
            )
        except ValidationError as e:
            raise ValueError(f"Invalid manifest: {e}") from e
        except OSError as e:
            raise type(e)(f"Error while loading manifest: {e}") from e

        chroma_deleted_chunks_count: int = len(self.delete_chunks_ids)

        # add missing extensions to the manifest
        self.manifest.extensions.extend(
            [ext for ext in self.extensions
             if ext not in self.manifest.extensions]
        )

        # collect corresponding files
        try:
            files: list[Path] = self._collect_files()
        except OSError as e:
            raise type(e)(f"Error while collecting files: {e}") from e
        self.lm.logger.debug("Found %d files", len(files))

        if not files:
            raise ValueError(
                f"No files found to index in {self.directory_path}"
            )

        # sync manifest files
        delete_chunks_ids, updated_files_ids, new_files_ids = (
            self.manifest.sync_files(files)
        )
        self.delete_chunks_ids.extend(delete_chunks_ids)
        self.updated_files_ids.update(updated_files_ids)
        self.new_files_ids.update(new_files_ids)

        # chunk files
        try:
            chunks_content, chunks_metadata, chunks_ids = (
                self._split_into_chunks(files)
            )
        except ValidationError as e:
            raise ValueError(f"Error while chunking: {e}") from e
        except OSError as e:
            raise type(e)(f"Error while chunking: {e}") from e

        if not chunks_ids:
            raise ValueError("No chunks generated from collected files")

        # save bm25 database
        self._bm25_index(
            chunks_content, [{"id": chunk_id} for chunk_id in chunks_ids]
        )
        self.manifest.add_store(chunks_metadata, chunks_ids, "bm25")
        self.lm.logger.debug("Saved BM25 index to '%s'", str(BM25_DIRECTORY))

        # save chroma database
        chroma_added_chunks_count: int = 0
        chroma_updated_chunks_count: int = 0
        chroma_chunks_content, chroma_chunks_ids = self._chroma_filter(
            chunks_content, chunks_metadata, chunks_ids
        )

        chroma_updated_chunks_count = self._count_updated_chroma_chunks(
            chunks_metadata, chroma_chunks_ids
        )
        chroma_added_chunks_count = (
            len(chroma_chunks_ids) - chroma_updated_chunks_count
        )

        if chroma_chunks_ids or self.delete_chunks_ids:
            self._chroma_index(chroma_chunks_content, chroma_chunks_ids)
            self.manifest.add_store(
                chunks_metadata, chroma_chunks_ids, "chroma"
            )

            if chroma_chunks_ids:
                self.lm.logger.debug(
                    "Saved Chroma index to '%s'", str(CHROMA_DIRECTORY)
                )
            else:
                self.lm.logger.debug(
                    "Deleted stale Chroma chunks from '%s'",
                    str(CHROMA_DIRECTORY)
                )
        else:
            self.lm.logger.debug("Skipped Chroma index")

        self.lm.logger.debug(
            "BM25 - indexed: %d",
            len(chunks_ids),
        )

        self.lm.logger.debug(
            "Chroma - deleted: %d, added: %d, updated: %d",
            chroma_deleted_chunks_count,
            chroma_added_chunks_count,
            chroma_updated_chunks_count,
        )

        # save chunks metadata
        self._store_chunks(chunks_metadata)
        self.lm.logger.debug(
            "Stored chunks content and metadata to '%s'",
            str(OUTPUT_DIRECTORY)
        )

        # save manifest
        self._store_manifest()
        self.lm.logger.debug(
            "Stored manifest file to '%s'", str(MANIFEST_PATH.parent)
        )
