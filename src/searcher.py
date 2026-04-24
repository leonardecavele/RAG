# standard imports
import json
from typing import Any

# extern imports
import bm25s
import chromadb
from sentence_transformers import SentenceTransformer
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


class Searcher():
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, query: str, lm: LoggerManager, k: int = 5
    ) -> None:
        self.lm = lm

        if not BM25_DIRECTORY.exists():
            raise FileNotFoundError(
                "TODO"
            )
        if not BM25_DIRECTORY.is_dir():
            raise FileNotFoundError(
                "TODO"
            )
        # maybe cannot access case, maybe do a helper func
        # eventually use path.resolve()
        self.retriever = bm25s.BM25.load(str(BM25_DIRECTORY), load_corpus=True)

        self.k = min(
            k, len(self.retriever.corpus) if self.retriever.corpus else 0
        )
        self.query = query

        #if CHROMA_DIRECTORY.exists():
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
        self.chroma_collection = self.chroma_client.get_collection(name="chunks")
        self.embedding_model = SentenceTransformer(LLM_MODEL)

    def search(self) -> MinimalSearchResults:
        self.lm.logger.debug("Searching %r with k=%d", self.query, self.k)

        query_tokens = bm25s.tokenize(self.query)
        results, scores = self.retriever.retrieve(query_tokens, k=self.k)

        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            chunks_metadata: dict[str, dict[str, Any]] = json.load(f)

        sources: list[MinimalSource] = []
        for result in results[0]:
            metadata = chunks_metadata[result["id"]]

            sources.append(
                MinimalSource(
                    file_path=metadata["file_path"],
                    first_character_index=metadata["first_character_index"],
                    last_character_index=metadata["last_character_index"],
                )
            )

        msr = MinimalSearchResults(
            question_id="manual",
            question=self.query,
            retrieved_sources=sources
        )
        print(msr)

        return msr
