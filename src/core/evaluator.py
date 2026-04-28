# standard
import json
from pathlib import Path
from typing import Any

# extern
from pydantic import ValidationError, validate_call
from rich.console import Console

# local
from ..schemas.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from ..utils.logger import LoggerManager
from ..defines import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_UNANSWERED_QUESTIONS_PATH,
)
from ..services.translator import Translator
from .searcher import Searcher


OVERLAP_THRESHOLD: float = 0.05


class _PassthroughTranslator(Translator):
    def __init__(self) -> None:
        pass

    def translate_to_english(self, text: str) -> str:
        return " ".join(text.split())


class Evaluator:
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, lm: LoggerManager, console: Console,
        student_answer_path: str, dataset_path: str,
        k: int = 5, max_context_length: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")

        self.lm = lm
        self.console = console
        self.student_answer_path = Path(student_answer_path)
        self.dataset_path = Path(dataset_path)
        self.k = k
        self.max_context_length = max_context_length

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {path}") from e

    def _load_dataset(self) -> RagDataset:
        try:
            return RagDataset.model_validate(
                self._load_json(self.dataset_path)
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid dataset format: {self.dataset_path}"
            ) from e

    def _load_student_results(self) -> StudentSearchResults:
        try:
            return StudentSearchResults.model_validate(
                self._load_json(self.student_answer_path)
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid student results format: "
                f"{self.student_answer_path}"
            ) from e

    def _unanswered_dataset_path(self) -> Path:
        path = Path(str(self.dataset_path).replace(
            "AnsweredQuestions",
            "UnansweredQuestions",
        ))

        if path.exists():
            return path

        return Path(DEFAULT_UNANSWERED_QUESTIONS_PATH)

    def _generate_student_results(self) -> None:
        dataset_path = self._unanswered_dataset_path()
        save_directory = self.student_answer_path.parent

        self.console.print(
            f"Student results file not found: {self.student_answer_path}"
        )
        self.console.print("Running search_dataset first")

        searcher = Searcher(
            lm=self.lm,
            console=self.console,
            embedding_model=None,
            translator=_PassthroughTranslator(),
            dataset_path=str(dataset_path),
            save_directory=str(save_directory),
            k=self.k,
        )

        searcher.search_dataset()

        generated_path = save_directory / dataset_path.name

        if generated_path.exists():
            self.student_answer_path = generated_path

    @staticmethod
    def _range_length(source: MinimalSource) -> int:
        return max(
            0,
            source.last_character_index - source.first_character_index,
        )

    def _limited_source(self, source: MinimalSource) -> MinimalSource:
        end = min(
            source.last_character_index,
            source.first_character_index + self.max_context_length,
        )

        return MinimalSource(
            file_path=source.file_path,
            first_character_index=source.first_character_index,
            last_character_index=end,
        )

    @staticmethod
    def _overlap_length(
        expected: MinimalSource,
        retrieved: MinimalSource,
    ) -> int:
        start = max(
            expected.first_character_index,
            retrieved.first_character_index,
        )
        end = min(
            expected.last_character_index,
            retrieved.last_character_index,
        )

        return max(0, end - start)

    def _source_found(
        self,
        expected: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        expected_length = self._range_length(expected)

        if expected_length == 0:
            return False

        for retrieved in retrieved_sources:
            retrieved = self._limited_source(retrieved)

            if expected.file_path != retrieved.file_path:
                continue

            overlap = self._overlap_length(expected, retrieved)
            ratio = overlap / expected_length

            if ratio >= OVERLAP_THRESHOLD:
                return True

        return False

    def _question_score(
        self,
        expected: AnsweredQuestion,
        retrieved_sources: list[MinimalSource],
    ) -> float:
        if not expected.sources:
            return 0.0

        found = 0

        for expected_source in expected.sources:
            if self._source_found(expected_source, retrieved_sources):
                found += 1

        return found / len(expected.sources)

    @staticmethod
    def _answered_questions(
        dataset: RagDataset,
    ) -> dict[str, AnsweredQuestion]:
        questions: dict[str, AnsweredQuestion] = {}

        for question in dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                questions[question.question_id] = question

        return questions

    def _recall_at_k(
        self,
        expected_questions: dict[str, AnsweredQuestion],
        student_results: StudentSearchResults,
        k: int,
    ) -> float:
        if not expected_questions:
            return 0.0

        total_score = 0.0

        for result in student_results.search_results:
            expected = expected_questions.get(result.question_id)

            if expected is None:
                continue

            retrieved_sources = result.retrieved_sources[:k]
            total_score += self._question_score(expected, retrieved_sources)

        return total_score / len(expected_questions)

    def evaluate(self) -> dict[str, float]:
        if not self.student_answer_path.exists():
            self._generate_student_results()

        dataset = self._load_dataset()
        student_results = self._load_student_results()
        expected_questions = self._answered_questions(dataset)

        recalls: dict[str, float] = {}

        for k in (1, 3, 5, 10):
            effective_k = min(k, self.k)
            recalls[f"recall@{k}"] = self._recall_at_k(
                expected_questions,
                student_results,
                effective_k,
            )

        total_questions = len(dataset.rag_questions)
        questions_with_sources = len(expected_questions)
        questions_with_student_sources = sum(
            1
            for result in student_results.search_results
            if result.retrieved_sources
        )

        self.console.print(f"Student data is valid: {True}")
        self.console.print(f"Total number of questions: {total_questions}")
        self.console.print(
            f"Total number of questions with sources: "
            f"{questions_with_sources}"
        )
        self.console.print(
            f"Total number of questions with student sources: "
            f"{questions_with_student_sources}"
        )
        self.console.print()
        self.console.print("Evaluation Results")
        self.console.print("=" * 40)
        self.console.print(f"Questions evaluated: {questions_with_sources}")

        for name, score in recalls.items():
            self.console.print(f"{name.capitalize()}: {score:.3f}")

        return recalls
