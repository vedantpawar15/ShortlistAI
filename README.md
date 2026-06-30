# RecruitAI

Offline AI Candidate Ranking System for recruiter workflows.

RecruitAI ingests local candidate and job-description datasets, builds semantic candidate profiles, scores candidate-job fit with transparent hybrid features, and writes a challenge-compatible `submission.csv`. It is fully open source and does not require paid APIs.

## What Works

- Local JSON, JSONL, CSV, and DOCX ingestion.
- Candidate profile normalization with career history, skills, education, certifications, languages, behavior signals, experience, and current role/company.
- Offline-first embedding wrapper for `BAAI/bge-small-en-v1.5` with disk caching.
- FAISS vector-store wrapper with a NumPy fallback for minimal environments.
- Deterministic lexical ranking baseline that works before model weights are downloaded.
- Weighted transparent scoring and recruiter-facing explanations.
- FastAPI and Streamlit entry points.
- `submission.csv` generation and validation.

## Tech Stack

- Python 3.12
- Pandas, NumPy, Pydantic
- Sentence Transformers and FAISS for full semantic retrieval
- RapidFuzz and spaCy for richer extraction when installed
- FastAPI, Streamlit, Plotly
- Typer CLI, Loguru logging

The code includes lightweight fallbacks for several optional runtime libraries so core tests and local ranking can still run in constrained offline environments.

## Data

Challenge files live in `data/`:

- `sample_candidates.json`
- `candidate_schema.json`
- `sample_submission.csv`
- `job_description.docx`

## Quick Start

```bash
python -m pip install -r requirements.txt
python main.py health
python main.py rank
python tools/validate_submission.py submission.csv
```

The ranking command writes:

- `submission.csv`
- `outputs/submission.csv`

## Offline Models

Download HuggingFace models once, then run offline from the local cache:

- Embedding: `BAAI/bge-small-en-v1.5`
- Cross encoder target: `BAAI/bge-reranker-base`

The deterministic lexical ranking path remains available when model weights are not present.

## Development

```bash
make install
make test
make lint
make api
make dashboard
```

## Docker

```bash
docker build -t recruitai .
docker run --rm -v "%cd%/data:/app/data" recruitai python main.py rank
```

## Project Layout

```text
main.py                 CLI entry point
app.py                  Streamlit entry point
src/data_loader.py      Dataset ingestion and schema inference
src/preprocessing.py    Semantic candidate profile generation
src/embedding.py        Offline model embeddings and cache
src/vector_store.py     FAISS/NumPy retrieval store
src/feature_engineering.py
src/scoring.py
src/ranking_engine.py
src/explainer.py
src/api.py
src/dashboard.py
tests/
tools/validate_submission.py
```
