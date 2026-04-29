*This project has been created as part of the 42 curriculum by ldecavel*.

```
data
├── datasets
│   ├── AnsweredQuestions
│   └── UnansweredQuestions
├── output
│   └── search_results
├── processed
│   ├── bm25
│   ├── chroma
│   ├── chunks_metadata.json
│   └── manifest.json
└── raw
    └── vllm-0.10.1
```

A README.md file must be provided at the root of your Git repository. Its purpose is
to allow anyone unfamiliar with the project (peers, staff, recruiters, etc.) to quickly
understand what the project is about, how to run it, and where to find more information
on the topic.
The README.md must include at least:
• The very first line must be italicized and read: This project has been created as part
of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].
• A “Description” section that clearly presents the project, including its goal and a
brief overview.
• An “Instructions” section containing any relevant information about compilation,
installation, and/or execution.
• A “Resources” section listing classic references related to the topic (documentation, articles, tutorials, etc.), as well as a description of how AI was used —
specifying for which tasks and which parts of the project.
➠ Additional sections may be required depending on the project (e.g., usage
examples, feature list, technical choices, etc.).
Any required additions will be explicitly listed below.
For this project, the README.md must also include:
• System architecture: Describe your RAG pipeline components and how they
interact
• Chunking strategy: Explain your approach to document segmentation
• Retrieval method: Detail the retrieval algorithm and ranking mechanism
• Performance analysis: Discuss recall@k scores and system performance
• Design decisions: Explain key implementation choices
20
RAG against the machine Will you answer my questions?
• Challenges faced: Document difficulties encountered and solutions
• Example usage: Provide clear examples of running your system
Your README must be written in English


 Query expansion (e.g., synonym expansion, query rewriting)
• Semantic embeddings for retrieval
• Result caching (index caching, query caching, etc.)
• Hybrid retrieval combining multiple methods
• Local LLM inference via vLLM

