# standard imports
import os
import sys

# extern imports
import fire

# local imports
from .error import ErrorCode
from .logger import Logger


class CLI:
    def __init__(self, log: str = "error") -> None:
        self.logger_manager: Logger = Logger(log)
        self.logger = self.logger_manager.logger

    def index(self, path: str, max_chunk_size: int = 2000) -> None:
        self.logger.debug(
            "Indexing %s with chunk size %d", path, max_chunk_size
        )

    def search(self, query: str, k: int = 5) -> None:
        self.logger.debug("Searching %r with k=%d", query, k)

    def answer(self, query: str, k: int = 5) -> None:
        self.logger.debug("Answering %r with k=%d", query, k)


def main() -> ErrorCode:
    os.environ.setdefault("PAGER", "cat")

    try:
        fire.Fire(CLI)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return ErrorCode.ARGS_ERROR

    return ErrorCode.NO_ERROR


if __name__ == "__main__":
    sys.exit(main().value)
