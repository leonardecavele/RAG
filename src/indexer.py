# standard imports
from pathlib import Path
from typing import Any

# extern imports
import bm25s
from langchain_core.documents import Document
from pydantic import (
    validate_call, PositiveInt, BaseModel, Field, ValidationError
)

# local imports
from .text_splitter import TextSplitter
from .logger import LoggerManager


class ChunkMetadata(BaseModel):
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


class Indexer:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, directory_path: str,
        lm: LoggerManager,
        chunk_size: PositiveInt = 2000
    ) -> None:
        self.output_directory = Path("data/processed")
        self.bm25_directory = self.output_directory / "bm25"

        self.directory_path = Path(directory_path)
        self.lm = lm
        self.chunk_size = chunk_size

        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"Path does not exist: {self.directory_path}"
            )

        if not self.directory_path.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {self.directory_path}"
            )

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []

        for path in self.directory_path.rglob("*"):
            if not path.is_file():
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
    ) -> tuple[list[str], list[dict[str, Any]]]:

        chunks_content: list[str] = []
        chunks_metadata: list[dict[str, Any]] = []

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
                    chunks_content.append(chunk.page_content)

                    metadata: dict[str, Any] = {
                        "file_path": str(file),
                        "first_character_index": index,
                        "last_character_index": (
                            index + len(chunk.page_content)
                        ),
                    }
                    chunks_metadata.append(
                        ChunkMetadata(**metadata).model_dump()
                    )
                    index += len(chunk.page_content)

            except (ValidationError, OSError) as e:
                raise type(e)(f"Error with {file}: {e}") from e

        return chunks_content, chunks_metadata

    def bm25_index(
        self, chunks_content: list[str], chunks_metadata: list[dict[str, Any]]
    ) -> None:
        self.bm25_directory.mkdir(parents=True, exist_ok=True)

        corpus_tokens = bm25s.tokenize(chunks_content)
        retriever = bm25s.BM25(corpus=chunks_metadata)
        retriever.index(corpus_tokens)
        retriever.save(str(self.bm25_directory))

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
            chunks_content, chunks_metadata = self._split_into_chunks(files)
        except (ValidationError, OSError) as e:
            raise type(e)(f"Error while chunking: {e}") from e
        self.lm.logger.debug(
            "Split documents into %d chunks", len(chunks_content)
        )

        self.bm25_index(chunks_content, chunks_metadata)
        self.lm.logger.debug(
            "Saved BM25 index to '%s'", str(self.bm25_directory)
        )
