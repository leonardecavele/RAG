# standard imports
import re
import json
from pathlib import Path
from typing import Any

# extern imports
import bm25s
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from pydantic import (
    validate_call, PositiveInt, BaseModel, Field, ValidationError
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
        self.content_path = self.output_directory / "chunks.json"

        self.lm = lm
        self.chunk_size = chunk_size

        self.extensions = self._parse_extensions(extensions)

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

        chunk_id: int = 0
        for file in files:
            try:
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

                    chunks_metadata.append(
                        ChunkMetadata(
                            **{
                                "content": content,
                                "file_path": str(file),
                                "first_character_index": index,
                                "last_character_index": (
                                    index + len(chunk.page_content)
                                ),
                            }
                        ).model_dump()
                    )
                    index += len(chunk.page_content)
                    chunks_ids["bm25"].append({"id": f"chunk_{chunk_id}"})
                    chunks_ids["chroma"].append(f"chunk_{chunk_id}")
                    chunk_id += 1

            except ValidationError as e:
                raise ValueError(f"Error with {file}: {e}") from e
            except OSError as e:
                raise type(e)(f"Error with {file}: {e}") from e

        return chunks_content, chunks_metadata, chunks_ids

    def bm25_index(
        self, chunks_content: list[str], chunks_ids: list[dict[str, str]]
    ) -> None:
        self.bm25_directory.mkdir(parents=True, exist_ok=True)

        corpus_tokens = bm25s.tokenize(chunks_content)
        retriever = bm25s.BM25(corpus=chunks_ids)
        retriever.index(corpus_tokens)
        retriever.save(str(self.bm25_directory))

    def chroma_index(
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

        for i in range(0, len(chunks_content), MAX_BATCH_SIZE):
            batch_content = chunks_content[i:i + MAX_BATCH_SIZE]
            batch_ids = chunks_ids[i:i + MAX_BATCH_SIZE]
            batch_embeddings = embedding_function(batch_content)
            collection.add(embeddings=batch_embeddings, ids=batch_ids)

    def store_chunks(self, chunks_metadata: list[dict[str, Any]]) -> None:
        with open(self.content_path, "w") as f:
            json.dump(chunks_metadata, f)

    def index_directory(self) -> None:
        self.lm.logger.debug(
            "Indexing %s with chunk size %d",
            str(self.directory_path), self.chunk_size
        )

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

        self.bm25_index(chunks_content, chunks_ids["bm25"])
        self.lm.logger.debug(
            "Saved BM25 index to '%s'", str(self.bm25_directory)
        )

        self.chroma_index(chunks_content, chunks_ids["chroma"])
        self.lm.logger.debug(
            "Saved Chroma index to '%s'", str(self.chroma_directory)
        )

        self.store_chunks(chunks_metadata)
        self.lm.logger.debug(
            "Stored chunks content and metadata to '%s'",
            str(self.output_directory)
        )
