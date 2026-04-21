# standard imports
from pathlib import Path
from typing import Any

# extern imports
from langchain_core.documents import Document
from pydantic import validate_call, PositiveInt, BaseModel, Field, ValidationError

# local imports
from .text_splitter import TextSplitter
from .logger import LoggerManager


class ChunkMetadata(BaseModel):
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


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


def create_chunks(
    files: list[Path], chunk_size: int, lm: LoggerManager
) -> tuple[list[str], list[dict[str, Any]]]:
    chunks_content: list[str] = []
    chunks_metadata: list[dict[str, Any]] = []

    for file in files:
        try:
            # load document
            doc = load_document(file)
            lm.logger.debug("Loaded '%s' file", str(file))

            # split into chunks
            splitter = TextSplitter.from_filename(
                str(file), chunk_size=chunk_size
            )
            file_chunks = splitter.split_documents([doc])

            # add metadata
            index: int = 0
            for chunk in file_chunks:
                chunks_content.append(chunk.page_content)
                metadata: dict[str, Any] = {
                    "file_path": str(file),
                    "first_character_index": index,
                    "last_character_index": index + chunk_size,
                }
                chunks_metadata.append(ChunkMetadata(**metadata).model_dump())
                index += chunk_size
        except (ValidationError, OSError) as e:
            raise type(e)(f"Error with {file}: {e}") from e

    return (chunks_content, chunks_metadata)


@validate_call
def index(
    path: str, lm: LoggerManager, chunk_size: PositiveInt = 2000
) -> None:
    lm.logger.debug("Indexing %s with chunk size %d", path, chunk_size)

    # collect files
    try:
        files = collect_files(Path(path))
    except OSError as e:
        raise type(e)(f"Error while collecting files: {e}") from e
    lm.logger.debug("Found %d files", len(files))

    # create chunks
    try:
        chunks_content, chunks_metadata = create_chunks(files, chunk_size, lm)
    except (ValidationError, OSError) as e:
        raise type(e)(f"Error while chunking: {e}") from e
    lm.logger.debug("Split documents into %d chunks", len(chunks_content))

    # 4. build a BM25 index with bm25s from chunk contents
    # 5. save chunks and bm25 index under data/processed/
