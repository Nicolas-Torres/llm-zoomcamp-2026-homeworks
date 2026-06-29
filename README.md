# LLM Zoomcamp 2026 — Homeworks
Solutions to the assignments from LLM Zoomcamp by DataTalks.Club, 2026 cohort (live cohort mode).

The course builds an end-to-end LLM application module by module: RAG, vector search, orchestration, evaluation, and monitoring. This repository contains the code for the weekly assignments. The final project lives in a separate repository.

Repository Structure
The project is organized as a uv workspace: a single lockfile and a shared environment at the root, with each assignment as an independent member.
```
llm-zoomcamp-2026-homeworks/
├── hw-1-agentic-RAG/        # Agentic RAG
│   └── main.py
├── hw_2_vector_search/      # Vector Search
│   ├── download.py
│   ├── embedder.py
│   └── main.py
├── homework-3/              # Orchestration
├── homework-4/              # Evaluation
├── homework-5/              # Monitoring
├── .python-version
├── pyproject.toml           # workspace root (members = ["..."])
├── uv.lock
├── .gitignore
└── README.md
```

# Prerequisites

- uv installed
- Python 3.12+ (uv can install it for you)
- An API key from an LLM provider (OpenAI by default) for assignments that call the model

# Initial Setup
Clone the repository and sync the full workspace:

```
git clone https://github.com/Nicolas-Torres/llm-zoomcamp-2026-homeworks.git
cd llm-zoomcamp-2026-homeworks
uv sync --all-packages
```

# Environment Variables

An .env file in the root folder.

```
OPENAI_API_KEY=sk-...
```

# How to Run Assignments

| Homework | Execution Path | Command to Run |
| :--- | :--- | :--- |
| **homework 1** | Project Root (`/`) | `uv run -m hw-1-agentic-RAG.main` |
| **homework 2** | Project Root (`/`) | `python hw_2_vector_search/download.py`<br>`uv run -m hw_2_vector_search.main` |

# Progress
| # | Module | Topic | Status |
|---|--------|------|--------|
| 1 | Agentic RAG | RAG from scratch, minsearch keyword search, chunking, agents with function calling | ✅ Completed |
| 2 | Vector Search | Embeddings, vector search, hybrid search | ✅ Completed |
| 3 | Orchestration | Ingestion pipelines and orchestration | ⬜ Pending |
| 4 | Evaluation | Retrieval metrics and LLM-as-a-Judge | ⬜ Pending |
| 5 | Monitoring | User feedback and dashboards | ⬜ Pending |