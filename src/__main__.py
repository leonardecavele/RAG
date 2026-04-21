# standard imports
import os
import sys
from pathlib import Path

# extern imports
import fire
from pydantic import (
    ValidationError, TypeAdapter, PositiveInt, BaseModel, Field
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# local imports
from .error import ErrorCode
from .logger import LoggerManager
from .text_splitter import TextSplitter


def load_document(path: Path) -> Document:
    content: str = path.read_text(encoding="utf-8", errors="ignore")  # errors?
    return Document(
        page_content=content,
        metadata={"source": str(path), "suffix": path.suffix}
    )


def collect_files(root: Path) -> list[Path]:
    allowed = {".py", ".md"}
    ignored = {".git", "__pycache__", ".venv", "venv"}

    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix in allowed:
            files.append(path)

    return files


class Chunk(BaseModel):
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)
    content: str


class CLI:
    def index(
        self, path: str, chunk_size: int = 2000, level: str = "error"
    ) -> None:
        path = TypeAdapter(str).validate_python(path)
        chunk_size = TypeAdapter(PositiveInt).validate_python(chunk_size)
        self._init_logger(level)
        self.logger.debug("Indexing %s with chunk size %d", path, chunk_size)

        # collect files
        files = collect_files(Path(path))
        self.logger.debug("Found %d files", len(files))

        chunks: list[Chunk] = []
        for file in files:
            try:
                doc = load_document(file)
                self.logger.debug("Loaded '%s' file", str(file))
                splitter = TextSplitter.from_filename(
                    str(file), chunk_size=chunk_size
                )

                file_chunks = splitter.split_documents([doc])
                index: int = 0
                for chunk in file_chunks:
                    c = Chunk(
                        file_path=str(file),
                        content=chunk.page_content,
                        first_character_index=index,
                        last_character_index=index + chunk_size
                    )
                    chunks.append(c)
                    index += chunk_size
            except Exception as e:
                raise type(e)(f"Error with {file}: {e}") from e
        self.logger.debug("Built %d chunks", len(chunks))

        # 0. discover useful repository files
        # 1. load documents with langchain loaders
        # 2. split documents with a strategy depending on file type
        #    - python code splitter
        #    - markdown/text splitter
        # 3. convert split documents into internal chunks with:
        #    file_path, first_character_index, last_character_index, content
        # 4. build a BM25 index with bm25s from chunk contents
        # 5. save chunks and bm25 index under data/processed/

    def search(
        self, query: str, k: int = 5, level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level)
        self.logger.debug("Searching %r with k=%d", query, k)

    def answer(
        self, query: str, k: int = 5, level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level)
        self.logger.debug("Answering %r with k=%d", query, k)

    def _init_logger(self, level: str) -> None:
        level = TypeAdapter(str).validate_python(level)
        self.lm: LoggerManager = LoggerManager(level)
        self.logger = self.lm.logger


def main() -> ErrorCode:
    os.environ.setdefault("PAGER", "cat")

    try:
        fire.Fire(CLI())
    except ValidationError as e:
        print(f"{type(e).__name__}: {e.errors()[0]['msg']}")
        return ErrorCode.ARGS_ERROR
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return ErrorCode.ARGS_ERROR

    return ErrorCode.NO_ERROR


if __name__ == "__main__":
    sys.exit(main().value)
