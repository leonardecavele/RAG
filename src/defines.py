from pathlib import Path

OUTPUT_DIRECTORY = Path("data/processed")
BM25_DIRECTORY = OUTPUT_DIRECTORY / "bm25"
CHROMA_DIRECTORY = OUTPUT_DIRECTORY / "chroma"

CHUNKS_METADATA_PATH = OUTPUT_DIRECTORY / "chunks_metadata.json"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"

MAX_BATCH_SIZE: int = 1024

EMBEDDING_MODEL: str = (
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # "intfloat/multilingual-e5-small"
    "all-MiniLM-L6-v2"
)
TRANSLATION_MODEL: str = (
    "Helsinki-NLP/opus-mt-mul-en"
)

DEFAULT_VLLM: str = "vllm-0.10.1"
