# standard
import uuid

# extern
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    content: str
    file_path: str
    first_character_index: int = Field(0, ge=0)
    last_character_index: int = Field(0, ge=0)


class CachedFile(BaseModel):
    stores: set[str] = Field(default_factory=set)
    file_path: str
    file_hash: str
    chunks_ids: set[str]


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
