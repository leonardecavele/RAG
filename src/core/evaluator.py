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


OVERLAP_THRESHOLD: float = 0.05


class Evaluator:
    """Evaluate retrieved sources against an answered dataset."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, lm: LoggerManager, console: Console,
        student_answer_path: str, dataset_path: str, k: int = 5
    ) -> None:
        """Initialize evaluator paths and recall limit."""

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.lm = lm
        self.console = console
        self.student_answer_path = Path(student_answer_path)
        self.dataset_path = Path(dataset_path)
        self.k = k

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load and validate a JSON file."""

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
        """Load the expected answer dataset."""

        try:
            return RagDataset.model_validate(
                self._load_json(self.dataset_path)
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid dataset format: {self.dataset_path}"
            ) from e

    def _load_student_results(self) -> StudentSearchResults:
        """Load student search results."""

        try:
            return StudentSearchResults.model_validate(
                self._load_json(self.student_answer_path)
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid student results format: "
                f"{self.student_answer_path}"
            ) from e

    @staticmethod
    def _range_length(source: MinimalSource) -> int:
        """Return the non-negative character range length."""

        return int(max(
            0, source.last_character_index - source.first_character_index
        ))

    @staticmethod
    def _overlap_length(
        expected: MinimalSource,
        retrieved: MinimalSource,
    ) -> int:
        """Return the overlap length between two source ranges."""

        start = max(
            expected.first_character_index,
            retrieved.first_character_index,
        )
        end = min(
            expected.last_character_index,
            retrieved.last_character_index,
        )

        return int(max(0, end - start))

    def _source_found(
        self, expected: MinimalSource, retrieved_sources: list[MinimalSource]
    ) -> bool:
        """Return whether an expected source overlaps a retrieved one."""

        expected_length = self._range_length(expected)

        if expected_length == 0:
            return False

        for retrieved in retrieved_sources:
            if expected.file_path != retrieved.file_path:
                continue

            retrieved_length = self._range_length(retrieved)

            if retrieved_length == 0:
                continue

            overlap = self._overlap_length(expected, retrieved)
            union_start = min(
                expected.first_character_index,
                retrieved.first_character_index,
            )
            union_end = max(
                expected.last_character_index,
                retrieved.last_character_index,
            )
            union_length = max(0, union_end - union_start)

            if union_length == 0:
                continue

            ratio = overlap / union_length

            if ratio >= OVERLAP_THRESHOLD:
                return True

        return False

    def _question_score(
        self,
        expected: AnsweredQuestion,
        retrieved_sources: list[MinimalSource],
    ) -> float:
        """Count expected sources found for one question."""

        if not expected.sources:
            return 0.0

        found = 0

        for expected_source in expected.sources:
            if self._source_found(expected_source, retrieved_sources):
                found += 1

        return float(found)

    @staticmethod
    def _answered_questions(
        dataset: RagDataset
    ) -> dict[str, AnsweredQuestion]:
        """Return answered dataset questions by ID."""

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
        """Calculate source recall at a cutoff."""

        total_expected_sources = sum(
            len(question.sources)
            for question in expected_questions.values()
        )

        if total_expected_sources == 0:
            return 0.0

        total_found = 0.0
        for result in student_results.search_results:
            expected = expected_questions.get(result.question_id)

            if expected is None:
                continue

            retrieved_sources = result.retrieved_sources[:k]
            total_found += self._question_score(expected, retrieved_sources)

        return total_found / total_expected_sources

    def evaluate(self) -> dict[str, float]:
        """Evaluate student results and print recall metrics."""

        dataset = self._load_dataset()
        student_results = self._load_student_results()
        expected_questions = self._answered_questions(dataset)

        recalls: dict[str, float] = {}

        max_k = min(self.k, student_results.k)
        for recall_k in (1, 3, 5, 10):
            if recall_k > max_k:
                continue

            recalls[f"recall@{recall_k}"] = self._recall_at_k(
                expected_questions,
                student_results,
                recall_k,
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
        self.console.print(f"Questions evaluated: {questions_with_sources}")

        for name, score in recalls.items():
            self.console.print(f"{name.capitalize()}: {score:.3f}")

        return recalls
