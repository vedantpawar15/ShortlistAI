# RecruitAI-Pro

## Objective

Build a hackathon-winning AI Candidate Ranking System.

The project must be completely open source.

No paid APIs.

Everything should work offline after downloading HuggingFace models.

---

## Current Status

Architecture Completed

Modules Present

✔ data_loader.py
✔ preprocessing.py
✔ embedding.py
✔ ranking_engine.py
✔ reranker.py
✔ feature_engineering.py
✔ scoring.py
✔ dashboard.py
✔ api.py

Tests

25 Passing

2 Failing

Failing Tests

1.

tests/test_preprocessing.py

Reason

Semantic profile generation is missing the complete career history string.

Expected

career history title company description

Current output omits part of the title.

2.

tests/test_data_loader.py

Schema inference mismatch.

Expected

object dtype

Current implementation needs investigation by tracing the failing test.

---

Datasets

sample_candidates.json

candidate_schema.json

sample_submission.csv

job_description.docx

All datasets are inside

data/

---

Architecture

Data Loader

↓

Preprocessing

↓

Embedding

↓

FAISS Retrieval

↓

Cross Encoder Reranking

↓

Hybrid Feature Engineering

↓

Weighted Scoring

↓

Explanation Engine

↓

submission.csv

---

Target Models

Embedding

BAAI/bge-small-en-v1.5

Cross Encoder

BAAI/bge-reranker-base

Vector Store

FAISS

Framework

FastAPI

Dashboard

Streamlit

---

Requirements

No placeholders

No TODOs

Production quality

Type hints

Logging

Unit tests

Documentation

README

Docker

GitHub ready

---

Goal

Produce a repository that passes every unit test and generates

submission.csv

from the provided datasets.