# RecruitAI

RecruitAI is an offline-first candidate ranking system for recruiter workflows. It ingests local challenge files, builds semantic candidate and job documents, ranks candidates with transparent weighted features, and writes a challenge-valid `submission.csv`.

## Current Pipeline

The live ranking flow is:

`DataLoader -> TextPreprocessor -> EmbeddingModel (optional) -> FaissVectorStore (optional) -> FeatureEngineer -> CandidateScorer -> Reranker -> RankingExplainer -> submission.csv`

Important behavior:

- If `sentence-transformers` and model weights are available, the engine can use `BAAI/bge-small-en-v1.5` embeddings and `BAAI/bge-reranker-base`.
- If those dependencies are missing, the system falls back to deterministic lexical ranking so the project still runs offline in constrained environments.
- FAISS is optional at runtime. A NumPy similarity fallback is implemented for minimal environments.

## What Works

- Local ingestion for JSON, JSONL, CSV, and DOCX files.
- Candidate semantic-profile generation from nested or flattened records.
- Job semantic-profile generation for retrieval and reranking.
- Embedding cache management for candidate and job documents.
- Retrieval with persisted item IDs and metadata.
- Transparent feature engineering, weighted scoring, and recruiter-facing explanations.
- FastAPI and Streamlit entry points.
- Submission generation and validation against the challenge format.

## Data Files

Challenge inputs in [data](E:/Coding/ShortlistAI/data):

- `sample_candidates.json`
- `candidate_schema.json`
- `sample_submission.csv`
- `job_description.docx`

## Setup

### Minimal test environment

Use this when you want `pytest` and the fallback ranking path only:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install pytest pandas numpy pydantic pydantic-settings
.venv\Scripts\python -m pytest -q
```

### Full project environment

Use this when you want embeddings, reranking, API, and dashboard dependencies:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Quick Start

```bash
.venv\Scripts\python main.py health
.venv\Scripts\python main.py rank
.venv\Scripts\python tools/validate_submission.py submission.csv
```

The ranking command writes:

- [submission.csv](E:/Coding/ShortlistAI/submission.csv)
- [outputs/submission.csv](E:/Coding/ShortlistAI/outputs/submission.csv)

## Models

Configured model targets:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker model: `BAAI/bge-reranker-base`

The code is written so these model-backed stages are optional at runtime. If the packages or weights are unavailable, the project still ranks candidates using the fallback feature-based path.

## Entrypoints

- CLI: [main.py](E:/Coding/ShortlistAI/main.py)
- FastAPI app: [src/api.py](E:/Coding/ShortlistAI/src/api.py)
- Streamlit app: [app.py](E:/Coding/ShortlistAI/app.py)
- Submission validator: [tools/validate_submission.py](E:/Coding/ShortlistAI/tools/validate_submission.py)

## Testing

```bash
.venv\Scripts\python -m pytest -q
```

## Project Layout

```text
main.py
app.py
src/data_loader.py
src/preprocessing.py
src/embedding.py
src/vector_store.py
src/feature_engineering.py
src/scoring.py
src/reranker.py
src/ranking_engine.py
src/explainer.py
src/api.py
src/dashboard.py
tests/
tools/validate_submission.py
```
