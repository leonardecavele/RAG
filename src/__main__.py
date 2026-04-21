# standard imports
import os
import sys

# extern imports
import fire
from pydantic import ValidationError, TypeAdapter, PositiveInt

# local imports
from .error import ErrorCode
from .logger import LoggerManager
from .indexer import Indexer


class CLI:
    def index(
        self, directory_path: str, chunk_size: int = 2000, level: str = "error"
    ) -> None:
        self._init_logger(level)
        try:
            i = Indexer(directory_path, self.lm, chunk_size)
        except (FileNotFoundError, NotADirectoryError, ValidationError) as e:
            raise type(e)(f"Error while indexing: {e}") from e
        i.index_directory()

    def search(
        self, query: str, k: int = 5, level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level)
        self.lm.logger.debug("Searching %r with k=%d", query, k)

    def answer(
        self, query: str, k: int = 5, level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level)
        self.lm.logger.debug("Answering %r with k=%d", query, k)

    def _init_logger(self, level: str) -> None:
        level = TypeAdapter(str).validate_python(level)
        self.lm: LoggerManager = LoggerManager(level)


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
