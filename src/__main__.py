# standard imports
import os
import sys

# extern imports
import fire
from rich.console import Console
from pydantic import ValidationError, TypeAdapter, PositiveInt

# local imports
from .error import ErrorCode
from .logger import LoggerManager
from .indexer import Indexer
from .searcher import Searcher
from .defines import DEFAULT_VLLM
from .display import print_msr


class CLI:
    def index(
        self, directory_path: str = DEFAULT_VLLM,
        chunk_size: int = 2000, extensions: str = "*", idiot: bool = False,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        try:
            i = Indexer(
                directory_path=directory_path,
                lm=self.lm,
                extensions=extensions,
                chunk_size=chunk_size,
                idiot=idiot,
            )
        except ValidationError as e:
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e
        try:
            i.index_directory()
        except ValidationError as e:
            raise ValueError(f"Error while indexing: {e}") from e
        except OSError as e:
            raise type(e)(f"Error while indexing: {e}") from e

    def search(
        self, query: str, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        self._init_console()
        try:
            s = Searcher(
                query=query,
                lm=self.lm,
                k=k
            )
        except ValidationError as e:
            raise ValueError(f"Error with the arguments: {e}") from e  # todo
        try:
            print_msr(s.search(), query)
        except ValidationError as e:
            raise ValueError(f"Error while indexing: {e}") from e  # to do

    def answer(
        self, query: str, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level, library_level)
        self.lm.logger.debug("Answering %r with k=%d", query, k)

    def _init_logger(self, level: str, library_level: str) -> None:
        level = TypeAdapter(str).validate_python(level)
        self.lm: LoggerManager = LoggerManager(level)
        self.lm.library_level(library_level)

    def _init_console(self) -> None:
        self.console = Console()


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
