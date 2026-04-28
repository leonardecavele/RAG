# standard
import os
import sys
import logging
import inspect

# extern
import fire
import torch
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# local
from .error import ErrorCode, print_validation_error, error_code
from .utils.logger import LoggerManager
from .core.indexer import Indexer
from .core.searcher import Searcher
from .core.answerer import Answerer
from .defines import (
    DEFAULT_VLLM, EMBEDDING_MODEL, DEFAULT_SAVE_DIRECTORY,
    DEFAULT_DATASET_PATH, CHROMA_DIRECTORY, LLM_MODEL, DEFAULT_RESULTS_PATH
)
from .display.results import print_msr
from .services.translator import Translator


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
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            print_msr(self.console, s.search(), query)
        except ValidationError as e:
            raise ValueError(f"Error while searching: {e}") from e
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
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            s.search_dataset()
        except ValidationError as e:
            raise ValueError(f"Error while searching: {e}") from e
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while searching: {e}") from e

    def answer(
        self, query: str, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        query = TypeAdapter(str).validate_python(query)
        k = TypeAdapter(PositiveInt).validate_python(k)

        self._init_logger(level, library_level)
        self._init_console()
        self._load_models()

        try:
            a = Answerer(
                query=query,
                lm=self.lm,
                console=self.console,
                embedding_model=self.embedding_model,
                translator=self.translator,
                tokenizer=self.tokenizer,
                llm_model=self.llm_model,
                k=k
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            a.answer()
        except ValidationError as e:
            raise ValueError(f"Error while answering: {e}") from e
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while answering: {e}") from e

    def answer_dataset(
        self, student_search_results_path: str = DEFAULT_RESULTS_PATH,
        save_directory: str = DEFAULT_SAVE_DIRECTORY, k: int = 5,
        level: str = "error", library_level: str = "error"
    ) -> None:
        self._init_logger(level, library_level)
        self._init_console()
        self._load_models()

        try:
            a = Answerer(
                lm=self.lm,
                console=self.console,
                embedding_model=self.embedding_model,
                translator=self.translator,
                tokenizer=self.tokenizer,
                llm_model=self.llm_model,
                dataset_path=student_search_results_path,
                save_directory=save_directory,
                k=k
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            a.answer_dataset()
        except ValidationError as e:
            raise ValueError(f"Error while answering: {e}") from e
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while answering: {e}") from e

    def _called_from(self, function_name: str) -> bool:
        frame = inspect.currentframe()

        while frame is not None:
            if frame.f_code.co_name == function_name:
                return True
            frame = frame.f_back

        return False

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
        self.tokenizer = None
        self.llm_model = None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if CHROMA_DIRECTORY.exists():
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device,
            )

        self.translator = Translator()

        if (
            self._called_from("answer")
            or self._called_from("answer_dataset")
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto",
            )
            self.llm_model.eval()


def main() -> ErrorCode:
    os.environ.setdefault("PAGER", "cat")

    try:
        fire.Fire(CLI())

    except ValidationError as e:
        print_validation_error(e)
        return ErrorCode.ARGS_ERROR

    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return ErrorCode.INTERRUPTED

    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return error_code(e)

    return ErrorCode.NO_ERROR


if __name__ == "__main__":
    sys.exit(main().value)
