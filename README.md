*This project has been created as part of the 42 curriculum by ldecavel*.

# RAG

## Description

This project is a local Retrieval-Augmented Generation system built to answer
questions about the vLLM codebase.

It indexes the repository, retrieves the most relevant source chunks for a
question, and can generate a short answer grounded in those retrieved sources.
The implementation follows the subject requirements while adding a few bonus
features such as semantic search, hybrid retrieval, Reciprocal Rank Fusion and
index caching.

## Features

- Indexes the vLLM repository from `data/raw/vllm-0.10.1`.
- Splits code and documentation into configurable chunks up to 2000 characters.
- Builds a BM25 lexical index.
- Builds a Chroma vector index with `all-MiniLM-L6-v2` embeddings.
- Combines BM25 and semantic results with weighted Reciprocal Rank Fusion.
- Translates non-English queries to English before retrieval.
- Generates answers locally with `Qwen/Qwen3-0.6B`.
- Evaluates retrieval quality with recall@k.
- Saves outputs using the Pydantic models required by the subject.

## Project Structure

```text
src/
├── core/        indexing, retrieval, answering and evaluation
├── display/     CLI result rendering
├── schemas/     Pydantic models and index manifest
├── services/    query translation
└── utils/       hashing, logging and text splitting

data/
├── raw/         vLLM source repository
├── datasets/    public answered and unanswered datasets
├── processed/   BM25 index, Chroma index, chunks and manifest
└── output/      generated search and answer files
```

## Instructions

The project uses `uv` and Python `3.13.13`.

Install dependencies:

```bash
make install
```

Index the default vLLM repository:

```bash
make run ARGS="index --max_chunk_size 2000"
```

Search one question:

```bash
make run ARGS='search "How does vLLM configure the OpenAI compatible server?" --k 5'
```

Answer one question:

```bash
make run ARGS='answer "How does vLLM configure the OpenAI compatible server?" --k 5'
```

Search a dataset:

```bash
make run ARGS="search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results \
  --k 5"
```

Generate answers from search results:

```bash
make run ARGS="answer_dataset \
  --student_search_results_path data/output/search_results/dataset_code_public.json \
  --save_directory data/output/search_results_and_answer"
```

Evaluate retrieval results:

```bash
make run ARGS="evaluate \
  --student_answer_path data/output/search_results/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 5"
```

Run static checks:

```bash
make lint
```

Clean generated caches:

```bash
make clean
```

## System Architecture

The pipeline is split into four main steps:

1. `Indexer` reads the vLLM repository, splits files into chunks and stores chunk
   metadata with file paths and character offsets.
2. The same chunks are indexed in BM25 for lexical matching and in Chroma for
   embedding-based semantic retrieval.
3. `Searcher` retrieves candidates from both indexes, merges them with weighted
   Reciprocal Rank Fusion, then returns the top-k sources.
4. `Answerer` rebuilds the context from the selected sources and asks
   `Qwen/Qwen3-0.6B` to produce a concise answer based only on that context.
5. `Evaluator` compares retrieved source ranges against the answered datasets and
   computes recall@k with the 5% overlap rule from the subject.

## Chunking Strategy

Chunking uses LangChain's recursive text splitter with language-aware separators.
Python, Markdown and many other source file extensions get dedicated separators;
unknown files fall back to generic recursive splitting.

Each chunk keeps:

- the original file path;
- `first_character_index`;
- `last_character_index`;
- the chunk content used to build prompts.

The default chunk size is `2000`, matching the subject limit, and can be changed
with `--max_chunk_size`.

Indexing is cached through `data/processed/manifest.json`: unchanged files keep
their Chroma embeddings, while removed or modified files invalidate their old
chunks. BM25 is rebuilt from the current collected files to keep the lexical
index simple and consistent.

## Retrieval Method

The mandatory retrieval baseline is BM25, implemented with `bm25s`.

The bonus retrieval path adds semantic search with Chroma and
`all-MiniLM-L6-v2`. For each query, the system retrieves a larger candidate set
from both stores, then merges rankings with Reciprocal Rank Fusion:

- BM25 weight: `0.65`;
- Chroma weight: `0.35`;
- RRF constant: `60`.

This keeps exact keyword and code-symbol matching strong while still using
semantic similarity for documentation-style questions.

## Design Decisions

- `uv` is used for reproducible dependency management.
- Python Fire provides the required CLI with simple command mapping.
- Pydantic validates datasets, search results, answers and cached metadata.
- BM25 remains the main retrieval signal because code questions often depend on
  exact names, paths and symbols.
- Semantic embeddings are added as a bonus signal instead of replacing BM25.
- Query translation improves retrieval for non-English questions against an
  English codebase.
- The local LLM prompt is short and source-grounded to reduce hallucinations.

## Challenges

- Keeping source offsets correct after chunking was important for evaluation, so
  each chunk is mapped back to its original character range.
- The vLLM repository is large, so indexing uses batching and persistent stores.
- Combining lexical and semantic search required weighting to avoid semantic
  matches hiding exact code-symbol matches.
- Local generation has limited context, so prompts are capped and answers are
  kept concise.

## Bonus Implemented

- Semantic embeddings for retrieval.
- Hybrid BM25 + Chroma retrieval.
- Weighted Reciprocal Rank Fusion ranking.
- Persistent index caching with manifest-based invalidation.
- Query translation before retrieval.

## Resources

- 42 subject: `en.subject.pdf`
- vLLM repository indexed in `data/raw/vllm-0.10.1`
- BM25: https://en.wikipedia.org/wiki/Okapi_BM25
- Chroma documentation: https://docs.trychroma.com/
- Sentence Transformers documentation: https://www.sbert.net/
- LangChain text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- Qwen models: https://huggingface.co/Qwen
- Python Fire: https://github.com/google/python-fire
- Pydantic documentation: https://docs.pydantic.dev/

AI was used as a support tool for README writing and project review: it helped
summarize the implemented architecture, compare it with the subject, and improve
the wording of the documentation.
