# standard imports
import os
import sys
import logging

# extern imports
import fire
from rich.console import Console
from rich.theme import Theme
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from pydantic import ValidationError, TypeAdapter, PositiveInt
from sentence_transformers import SentenceTransformer

# local imports
from .error import ErrorCode
from .logger import LoggerManager
from .indexer import Indexer
from .searcher import Searcher
from .defines import (
    DEFAULT_VLLM, EMBEDDING_MODEL, DEFAULT_SAVE_DIRECTORY,
    DEFAULT_DATASET_PATH, CHROMA_DIRECTORY
)
from .display import print_msr
from .translate import Translator


class CLI:
    def index(
        self, directory_path: str = DEFAULT_VLLM,
        chunk_size: int = 2000, extensions: str = "*", idiot: bool = False,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        self._init_console()
        try:
            i = Indexer(
                directory_path=directory_path,
                lm=self.lm,
                console=self.console,
                extensions=extensions,
                chunk_size=chunk_size,
                idiot=idiot,
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e
        try:
            i.index_directory()
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error while indexing: {e}") from e
        except OSError as e:
            raise type(e)(f"Error while indexing: {e}") from e

    def search(
        self, query: str, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        self._init_console()
        self._load_models()

        try:
            s = Searcher(
                query=query,
                lm=self.lm,
                console=self.console,
                embedding_model=self.embedding_model,
                translator=self.translator,
                k=k
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e  # todo
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            print_msr(self.console, s.search(), query)
        except ValidationError as e:
            raise ValueError(f"Error while searching: {e}") from e  # to do
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while searching: {e}") from e

    def search_dataset(
        self, dataset_path: str = DEFAULT_DATASET_PATH,
        save_directory: str = DEFAULT_SAVE_DIRECTORY, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        self._init_console()
        self._load_models()
        try:
            s = Searcher(
                lm=self.lm,
                console=self.console,
                embedding_model=self.embedding_model,
                translator=self.translator,
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e  # todo
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            s.search_dataset()
        except ValidationError as e:
            raise ValueError(f"Error while searching: {e}") from e  # to do
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while searching: {e}") from e

    def answer(
        self, query: str, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)
        self._init_logger(level, library_level)
        self.lm.logger.debug("Answering %r with k=%d", query, k)

    def _should_show_loader(self) -> bool:
        return (
            self.console.is_terminal
            and not self.lm.logger.isEnabledFor(logging.INFO)
        )

    def _load_models(self) -> None:
        if not self._should_show_loader():
            self._init_models()
            return

        with Progress(
            SpinnerColumn("shark", style="cyan"),
            TextColumn("[black]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            progress.add_task("Loading models", total=None)
            self._init_models()

    def _init_logger(self, level: str, library_level: str) -> None:
        level = TypeAdapter(str).validate_python(level)
        self.lm: LoggerManager = LoggerManager(level)
        self.lm.library_level(library_level)

    def _init_console(self) -> None:
        self.console = Console(
            theme=Theme({
                "progress.elapsed": "red",
                "progress.remaining": "black"
            })
        )

    def _init_models(self) -> None:
        self.embedding_model = None
        if CHROMA_DIRECTORY.exists():
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.translator = Translator()


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
