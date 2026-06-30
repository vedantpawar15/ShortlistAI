# RecruitAI

Production-grade offline AI Candidate Ranking System scaffold for recruiter workflows.

This repository is intentionally architecture-only at this stage. It defines the module boundaries, public classes, and application entry points for an offline ranking system built with open-source Python tooling.

## Goals

- Work fully offline with local datasets and local models.
- Avoid paid APIs and hosted vector databases.
- Support recruiter-facing ranking, explanations, dashboards, and API access.
- Keep the codebase modular enough for testing, observability, and later production hardening.

## Tech Stack

- Python 3.12
- Sentence Transformers
- FAISS
- RapidFuzz
- spaCy
- Pandas
- NumPy
- Scikit Learn
- FastAPI
- Streamlit
- Plotly
- Pydantic
- Loguru
- Typer

## Project Structure

```text
RecruitAI/
README.md
requirements.txt
.gitignore
Dockerfile
Makefile
LICENSE
config.py
main.py
app.py
data/
outputs/
logs/
models/
tests/
src/
```

## Current Status

The project architecture is present and import-safe. Core ranking functionality is intentionally not implemented yet.

## Common Commands

```bash
make install
make test
make lint
make api
make dashboard
```

