# extern imports
from pydantic import (
    validate_call, PositiveInt, BaseModel,
    Field, ValidationError, ConfigDict
)

# local imports
from .logger import LoggerManager
from .defines import (
    OUTPUT_DIRECTORY, BM25_DIRECTORY, CHROMA_DIRECTORY, CHUNKS_METADATA_PATH,
    MANIFEST_PATH, MAX_BATCH_SIZE, LLM_MODEL
)


class Searcher():
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, query: str, lm: LoggerManager, k: int = 5
    ) -> None:
        self.lm = lm

        self.query = query
        self.k = k

    def search(self) -> None:
        self.lm.logger.debug("Searching %r with k=%d", self.query, self.k)
