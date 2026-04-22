# standard imports
import re
import json
import hashlib
from pathlib import Path
from typing import Any

# extern imports
import bm25s
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from pydantic import (
    validate_call, PositiveInt, BaseModel, Field, ValidationError, ConfigDict
)

# local imports
from .text_splitter import TextSplitter
from .logger import LoggerManager


MAX_BATCH_SIZE: int = 5000


class ChunkMetadata(BaseModel):
    content: str
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


class CachedFile(BaseModel):
    file_path: str
    file_hash: str
    chunks_ids: set[str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: PositiveInt = 0
    extensions: list[str] = Field(default_factory=lambda: [])
    files_by_extensions: dict[str, dict[str, CachedFile]] = Field(
        default_factory=lambda: {}
    )

    @classmethod
    def from_file(cls, file_path: Path) -> "Manifest":
        manifest_data = {}

        try:
            if file_path.exists():
                with open(file_path, "r") as f:
                    manifest_data = json.load(f)
                if not isinstance(manifest_data, dict):
                    pass
                    # to do
        except json.JSONDecodeError as e:
            raise e from e  # to do
        return cls(**manifest_data)


class Indexer:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, directory_path: str, lm: LoggerManager,
        extensions: str = "*", chunk_size: PositiveInt = 2000
    ) -> None:
        self.directory_path = Path(directory_path)

        self.output_directory = Path("data/processed")
        self.bm25_directory = self.output_directory / "bm25"
        self.chroma_directory = self.output_directory / "chroma"
        self.chunks_metadata_path = self.output_directory / "chunks.json"
        self.manifest_path = self.output_directory / "manifest.json"

        self.lm = lm
        self.chunk_size = chunk_size

        self.extensions = self._parse_extensions(extensions)
        self.delete_chunks_ids: list[str] = []

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
        parsed_extensions: set[str] = set()

        for extension in extensions.split(":"):
            e: str = re.sub(r"[^a-zA-Z0-9]", "", extension).lower()
            if e:
                parsed_extensions.add(e)

        return parsed_extensions

    @staticmethod
    def _md5sum(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_md5sum(file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []

        for path in self.directory_path.rglob("*"):
            if not path.is_file():
                continue
            extension: str = path.suffix.removeprefix(".").lower()
            if extension not in self.extensions:
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
    ) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:

        chunks_content: list[str] = []
        chunks_metadata: list[dict[str, Any]] = []
        chunks_ids: dict[str, Any] = {"bm25": [], "chroma": []}

        for e in self.extensions:
            if e not in self.manifest.extensions:
                self.manifest.extensions.append(e)

        for file in files:
            file_id: str = self._md5sum(str(file))
            file_hash: str = self._file_md5sum(file)
            file_suffix: str = file.suffix

            manifest_files = self.manifest.files_by_extensions.setdefault(
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

            if (
                manifest_file.file_hash is not None
                and manifest_file.file_hash != file_hash
            ):
                self.delete_chunks_ids.extend(
                    manifest_file.chunks_ids
                )
                manifest_file.chunks_ids = set()

            manifest_file.file_path = str(file)
            manifest_file.file_hash = file_hash

            empty_manifest_ids: bool = False
            if not manifest_file.chunks_ids:
                empty_manifest_ids = True

            # load document
            doc: Document = self._load_document(file)
            self.lm.logger.debug("Loaded '%s' file", str(file))
            # split into chunks
            splitter: TextSplitter = TextSplitter.from_filename(
                str(file), chunk_size=self.chunk_size,
            )
            file_chunks: list[Document] = splitter.split_documents([doc])

            # add metadata
            index: int = 0
            for chunk in file_chunks:
                content: str = chunk.page_content
                chunks_content.append(content)

                last_character_index: int = index + len(content)

                chunk_id: str = (
                    "chunk"
                    f"_{file_id}"
                    f"_{index}"
                    f"_{last_character_index}"
                )
                if empty_manifest_ids:
                    manifest_file.chunks_ids.add(chunk_id)

                chunks_metadata.append(
                    ChunkMetadata(
                        **{
                            "content": content,
                            "file_path": str(file),
                            "first_character_index": index,
                            "last_character_index": last_character_index,
                        }
                    ).model_dump()
                )
                index += len(chunk.page_content)
                chunks_ids["bm25"].append({"id": f"{chunk_id}"})
                # control this
                if empty_manifest_ids:
                    chunks_ids["chroma"].append(f"{chunk_id}")

        return chunks_content, chunks_metadata, chunks_ids

    def _bm25_index(
        self, chunks_content: list[str], chunks_ids: list[dict[str, str]]
    ) -> None:
        self.bm25_directory.rmdir()
        self.bm25_directory.mkdir(parents=True, exist_ok=True)

        corpus_tokens = bm25s.tokenize(chunks_content)
        retriever = bm25s.BM25(corpus=chunks_ids)
        retriever.index(corpus_tokens)
        retriever.save(str(self.bm25_directory))

    def _chroma_index(
        self, chunks_content: list[str], chunks_ids: list[str]
    ) -> None:
        self.chroma_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_directory))

        embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        collection = client.get_or_create_collection(name="chunks")

        if self.delete_chunks_ids:
            for i in range(0, len(self.delete_chunks_ids), MAX_BATCH_SIZE):
                batch_delete_ids = self.delete_chunks_ids[i:i + MAX_BATCH_SIZE]
                collection.delete(ids=batch_delete_ids)

        for i in range(0, len(chunks_content), MAX_BATCH_SIZE):
            batch_content = chunks_content[i:i + MAX_BATCH_SIZE]
            batch_ids = chunks_ids[i:i + MAX_BATCH_SIZE]
            batch_embeddings = embedding_function(batch_content)
            collection.add(embeddings=batch_embeddings, ids=batch_ids)

    def _store_chunks(self, chunks_metadata: list[dict[str, Any]]) -> None:
        self.chunks_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.chunks_metadata_path, "w") as f:
            json.dump(chunks_metadata, f)

    def _store_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.model_dump(mode="json"), f, indent=4)

    def _load_manifest(self) -> None:
        self.manifest = Manifest.from_file(self.manifest_path)

        if self.manifest.chunk_size != self.chunk_size:
            for files in self.manifest.files_by_extensions.values():
                for file in files.values():
                    self.delete_chunks_ids.extend(file.chunks_ids)

            self.manifest = Manifest()
            return

        for e in self.manifest.files_by_extensions.copy():
            if e not in self.extensions:
                for file in self.manifest.files_by_extensions[e].values():
                    self.delete_chunks_ids.extend(file.chunks_ids)

                del self.manifest.files_by_extensions[e]

                if e in self.manifest.extensions:
                    self.manifest.extensions.remove(e)

                continue

            for file_id in self.manifest.files_by_extensions[e].copy():
                file = self.manifest.files_by_extensions[e][file_id]
                path = self.directory_path / file.file_path

                if not path.exists():
                    self.delete_chunks_ids.extend(file.chunks_ids)
                    del self.manifest.files_by_extensions[e][file_id]

            if not self.manifest.files_by_extensions[e]:
                del self.manifest.files_by_extensions[e]

                if e in self.manifest.extensions:
                    self.manifest.extensions.remove(e)

    def index_directory(self) -> None:
        self.lm.logger.debug(
            "Indexing %s with chunk size %d",
            str(self.directory_path), self.chunk_size
        )

        try:
            self._load_manifest()
        except Exception as e:  # to do
            raise e

        try:
            files: list[Path] = self._collect_files()
        except OSError as e:
            raise type(e)(f"Error while collecting files: {e}") from e

        self.lm.logger.debug("Found %d files", len(files))

        try:
            chunks_content, chunks_metadata, chunks_ids = (
                self._split_into_chunks(files)
            )
        except ValidationError as e:
            raise ValueError(f"Error while chunking: {e}") from e
        except OSError as e:
            raise type(e)(f"Error while chunking: {e}") from e
        self.lm.logger.debug(
            "Split documents into %d chunks", len(chunks_content)
        )

        self._bm25_index(chunks_content, chunks_ids["bm25"])
        self.lm.logger.debug(
            "Saved BM25 index to '%s'", str(self.bm25_directory)
        )

        self._chroma_index(chunks_content, chunks_ids["chroma"])
        self.lm.logger.debug(
            "Saved Chroma index to '%s'", str(self.chroma_directory)
        )

        self._store_chunks(chunks_metadata)
        self.lm.logger.debug(
            "Stored chunks content and metadata to '%s'",
            str(self.output_directory)
        )

        self._store_manifest()
        self.lm.logger.debug(
            "Stored manifest file to '%s'", str(self.manifest_path.parent)
        )
