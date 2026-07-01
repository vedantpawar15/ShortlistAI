# RecruitAI

RecruitAI is an interview-grade candidate ranking system for AI hiring workflows. The repository combines semantic search, BM25-style lexical retrieval, hybrid scoring, explicit feature attributions, a local API, a CLI demo, and a Streamlit dashboard.

## Why this implementation is stronger than a hackathon prototype

- Architecture is modular: domain models, retrieval, scoring, orchestration, API, dashboard, and tests are separated cleanly.
- Ranking is not a single heuristic. It fuses semantic similarity, lexical relevance, required and preferred skill coverage, experience fit, title alignment, location fit, profile freshness, and quantified achievements.
- Explainability is first-class. Every ranked result includes matched skills, missing skills, feature scores, feature contributions, and human-readable rationale lines.
- The system is dependency-light in core logic, making it easier to understand, test, and extend during interviews.

## Repository layout

- `main.py`: CLI demo and optional WSGI API server.
- `app.py`: Streamlit dashboard.
- `src/recruit_ai/domain.py`: candidate and job schemas.
- `src/recruit_ai/nlp.py`: tokenization, TF-IDF retrieval, BM25 scoring, reciprocal rank fusion.
- `src/recruit_ai/scoring.py`: hybrid scoring and explanations.
- `src/recruit_ai/pipeline.py`: orchestration layer.
- `src/recruit_ai/api.py`: local HTTP API.
- `src/recruit_ai/data/sample_data.py`: sample dataset for demos and tests.
- `tests/`: `unittest` suite.
- `docs/final_report.md`: engineering review and improvement report.

## Run

Use a Python 3.12+ environment.

```bash
python main.py
python main.py --json
python main.py --serve-api --port 8000
streamlit run app.py
python -m unittest discover -s tests -v
```

Install dashboard dependencies when needed:

```bash
pip install -r requirements.txt
```

## API

- `GET /health`
- `GET /sample`
- `POST /rank`

Example payload for `POST /rank`:

```json
{
  "job": {
    "job_id": "job-ml-001",
    "title": "Senior Applied AI Engineer",
    "summary": "Build retrieval-augmented ranking systems.",
    "required_skills": ["Python", "NLP", "Information Retrieval"],
    "preferred_skills": ["Streamlit", "Ranking"],
    "minimum_years_experience": 5,
    "location": "Remote - India",
    "responsibilities": ["Design ranking systems"],
    "keywords": ["semantic search"]
  },
  "candidates": []
}
```

## Extension ideas

- Replace TF-IDF with embedding-backed ANN retrieval.
- Add offline ranking evaluation metrics such as NDCG and precision@k from labeled judgments.
- Add fairness diagnostics and configurable de-biasing constraints.
- Add persistence, authentication, and async job handling for multi-tenant production use.
