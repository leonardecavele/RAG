# standard imports
import json
from typing import Any

# extern imports
import bm25s
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
from .types import MinimalSearchResults, MinimalSource
from .hash import md5sum


class Searcher():
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, query: str, lm: LoggerManager, k: int = 5
    ) -> None:
        self.lm = lm

        self.query = query
        self.k = k

    def search(self) -> MinimalSearchResults:
        self.lm.logger.debug("Searching %r with k=%d", self.query, self.k)

        retriever = bm25s.BM25.load(str(BM25_DIRECTORY), load_corpus=True)
        query_tokens = bm25s.tokenize(self.query)
        results, scores = retriever.retrieve(query_tokens, k=self.k)



        # recuperer les minimal sources
        # produire un minimal search results
        # output le minimal search results
