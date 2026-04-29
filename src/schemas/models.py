# standard
import uuid

# extern
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Store text and source offsets for an indexed chunk."""

    content: str
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


class CachedFile(BaseModel):
    """Track indexed stores and chunks for one source file."""

    stores: set[str] = Field(default_factory=set)
    file_path: str
    file_hash: str
    chunks_ids: set[str]


class MinimalSource(BaseModel):
    """Identify a retrieved source span in a file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources plus a generated answer."""

    answer: str


class UnansweredQuestion(BaseModel):
    """Represent a dataset question without expected sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a dataset question with expected sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent the RAG evaluation dataset."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """Represent student retrieval results for a dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent student retrieval results with generated answers."""

    search_results: list[MinimalAnswer]
