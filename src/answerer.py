# standard imports
import json
import logging
from queue import Empty, Queue
from threading import Thread
from typing import Any, Generator

# extern imports
import torch
from pydantic import validate_call, ValidationError
from rich.console import Console
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from transformers import TextIteratorStreamer

# local imports
from .logger import LoggerManager
from .types import MinimalAnswer, MinimalSearchResults
from .defines import (
    DEFAULT_DATASET_PATH, DEFAULT_SAVE_DIRECTORY, CHUNKS_METADATA_PATH
)
from .translate import Translator
from .searcher import Searcher


MAX_INPUT_TOKENS: int = 4096
MAX_NEW_TOKENS: int = 512
STREAMER_TIMEOUT: float = 1.0


class Answerer:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, lm: LoggerManager, console: Console, embedding_model: Any,
        translator: Translator, tokenizer: Any, llm_model: Any,
        query: str = "", k: int = 5,
        dataset_path: str = DEFAULT_DATASET_PATH,
        save_directory: str = DEFAULT_SAVE_DIRECTORY,
        question_id: str = "",
    ) -> None:
        self.lm = lm
        self.console = console
        self.embedding_model = embedding_model
        self.translator = translator
        self.tokenizer = tokenizer
        self.llm_model = llm_model

        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not CHUNKS_METADATA_PATH.exists():
            raise FileNotFoundError("chunks metadata file does not exist")
        if not CHUNKS_METADATA_PATH.is_file():
            raise FileNotFoundError("chunks metadata path is not a file")
        if self.tokenizer is None or self.llm_model is None:
            raise ValueError("LLM model is not loaded")

        self.query = query
        self.dataset_path = dataset_path
        self.save_directory = save_directory
        self.k = k
        self.question_id = question_id

    def _should_show_progress(self) -> bool:
        return (
            self.console.is_terminal
            and not self.lm.logger.isEnabledFor(logging.INFO)
        )

    def _context(self, msr: MinimalSearchResults) -> str:
        contexts: list[str] = []
        total = len(msr.retrieved_sources) + 1
        show_progress = self._should_show_progress()

        with Progress(
            SpinnerColumn("shark", style="cyan"),
            TextColumn("[black]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            disable=not show_progress,
        ) as progress:
            task_id = progress.add_task(
                "Loading chunks metadata", total=total
            )

            try:
                with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                    chunks_metadata: dict[str, dict[str, Any]] = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid metadata JSON file: {CHUNKS_METADATA_PATH}"
                ) from e
            except OSError as e:
                raise type(e)(
                    f"Error while reading chunks metadata: "
                    f"{CHUNKS_METADATA_PATH}"
                ) from e

            metadata_by_source: dict[tuple[str, int, int], dict[str, Any]] = {}

            for md in chunks_metadata.values():
                try:
                    key = (
                        md["file_path"],
                        md["first_character_index"],
                        md["last_character_index"],
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Invalid chunk metadata: missing {e}"
                    ) from e

                metadata_by_source[key] = md

            progress.advance(task_id)

            for index, source in enumerate(msr.retrieved_sources, start=1):
                progress.update(
                    task_id,
                    description=(
                        f"Building context "
                        f"{index}/{len(msr.retrieved_sources)}"
                    ),
                )

                key = (
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index,
                )
                md = metadata_by_source.get(key)

                if md is None:
                    raise ValueError(
                        f"Missing metadata for source: {source.file_path}"
                    )

                try:
                    snippet = str(md["content"])
                except KeyError as e:
                    raise ValueError(
                        f"Invalid chunk metadata for source "
                        f"{source.file_path}: missing {e}"
                    ) from e

                contexts.append(
                    "\n".join([
                        f"Source {index}:",
                        f"file_path: {source.file_path}",
                        f"first_character_index: "
                        f"{source.first_character_index}",
                        f"last_character_index: "
                        f"{source.last_character_index}",
                        "content:",
                        snippet,
                    ])
                )

                progress.advance(task_id)

        return "\n\n".join(contexts)

    def _input_device(self) -> Any:
        return next(self.llm_model.parameters()).device

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        return (
            isinstance(error, torch.OutOfMemoryError)
            or (
                isinstance(error, RuntimeError)
                and "out of memory" in str(error).lower()
            )
        )

    def _raise_generation_error(self, error: BaseException) -> None:
        if self._is_cuda_oom(error):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(
                "CUDA out of memory while generating the answer. "
                "Try reducing -k, reducing chunk_size, reducing "
                "MAX_INPUT_TOKENS, or using a smaller/quantized model."
            ) from error

        raise RuntimeError(f"Error while generating answer: {error}") from error

    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        if not context.strip():
            yield "I could not find enough information to answer."
            return

        system_prompt = (
            "You are an expert AI assistant. Answer the user's question "
            "based ONLY on the provided context below. If the answer cannot "
            "be found in the context, state it clearly without making up "
            "information. Do not include reasoning, analysis, or hidden "
            "thoughts. Mention the relevant source file when it supports "
            "the answer."
        )

        user_prompt = "\n".join([
            "Context:", context, "", "Question:", query
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_tokens_count: int = int(inputs["input_ids"].shape[-1])

        if input_tokens_count > MAX_INPUT_TOKENS:
            raise ValueError(
                "Prompt is too large for safe local generation: "
                f"{input_tokens_count} tokens, max is {MAX_INPUT_TOKENS}. "
                "Try reducing -k or chunk_size."
            )

        inputs = inputs.to(self._input_device())

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=STREAMER_TIMEOUT,
        )

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.3,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        generation_errors: Queue[BaseException] = Queue()

        def _generate() -> None:
            try:
                with torch.inference_mode():
                    self.llm_model.generate(**generation_kwargs)
            except BaseException as e:
                generation_errors.put(e)
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_generate, daemon=True)
        thread.start()

        while True:
            try:
                new_text = next(streamer)
            except StopIteration:
                break
            except Empty:
                if not thread.is_alive():
                    break
                continue

            yield new_text

        thread.join()

        if not generation_errors.empty():
            self._raise_generation_error(generation_errors.get())

    def _generate(self, query: str, context: str) -> str:
        answer = ""
        show_progress = self._should_show_progress()

        if not show_progress:
            for chunk in self.generate_answer(query, context):
                answer += chunk

            return answer.strip()

        with Live(
            Text("", style="white"),
            console=self.console,
            refresh_per_second=12,
            transient=False,
        ) as live:
            for chunk in self.generate_answer(query, context):
                answer += chunk
                live.update(Text(answer, style="white"))

        return answer.strip()

    def answer(self) -> MinimalAnswer:
        self.lm.logger.debug("Answering %r with k=%d", self.query, self.k)

        try:
            s = Searcher(
                query=self.query,
                lm=self.lm,
                console=self.console,
                embedding_model=self.embedding_model,
                translator=self.translator,
                k=self.k,
                question_id=self.question_id,
            )
        except (ValidationError, ValueError) as e:
            raise ValueError(f"Error with the arguments: {e}") from e
        except (FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error with the arguments: {e}") from e

        try:
            msr: MinimalSearchResults = s.search(
                show_progress=self._should_show_progress()
            )
        except ValidationError as e:
            raise ValueError(f"Error while searching: {e}") from e
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            raise type(e)(f"Error while searching: {e}") from e

        context = self._context(msr)
        query = self.translator.translate_to_english(self.query)

        self.lm.logger.debug("Query:\n%s", query)
        self.lm.logger.debug("Context:\n%s", context)

        try:
            answer = self._generate(query, context)
        except (ValueError, RuntimeError) as e:
            raise type(e)(f"Error while generating answer: {e}") from e

        if not self._should_show_progress():
            self.console.print(answer)

        return MinimalAnswer(
            question_id=msr.question_id,
            question=self.query,
            retrieved_sources=msr.retrieved_sources,
            answer=answer,
        )
