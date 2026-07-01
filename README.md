# ShortlistAI

[![Tests](https://img.shields.io/badge/tests-129%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#quick-start)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

An **offline-first AI candidate ranking system** for recruiter workflows. It ingests local challenge files, builds semantic candidate and job profiles, ranks candidates with a transparent hybrid scoring pipeline, and writes a challenge-valid `submission.csv`.

---

## Pipeline Architecture

```
DataLoader
  └── TextPreprocessor  (unicode → lowercase → punctuation → skill normalization)
        └── EmbeddingModel  (BAAI/bge-small-en-v1.5, disk-cached)
              └── FaissVectorStore  (semantic top-K retrieval)
                    └── FeatureEngineer
                          ├── semantic_similarity      (embedding cosine,   30%)
                          ├── skill_overlap            (required coverage,  25%)
                          ├── experience_match         (years vs req,       15%)
                          ├── education_match          (degree relevance,    5%)
                          ├── title_similarity         (current vs target,  10%)
                          ├── location_match           (geographic affinity, 5%)
                          └── behavioral_signal_score  (signals, GitHub,    10%)
                                └── CandidateScorer  (weighted sum + hybrid blend)
                                      └── Reranker  (cross-encoder or RRF fusion)
                                            └── RankingExplainer  (batch-calibrated)
                                                  └── submission.csv
```

### Hybrid Scoring

The pipeline blends two complementary ranking signals:

1. **Semantic retrieval** — BGE embedding cosine similarity via FAISS, stored as `semantic_similarity`.
2. **Feature scoring** — multi-signal weighted sum across skill, experience, education, title, location, and behavior dimensions.

```
final_score = α × semantic_similarity + (1 − α) × feature_score
```

Default `α = 0.6` (configurable at `RankingEngine(hybrid_alpha=...)` construction).

**Reciprocal Rank Fusion (RRF)** (`k=60`, Cormack et al. 2009) merges the feature-score rank list and the reranker rank list into the final ordering, providing diversity without requiring a cross-encoder.

### Score Dimensions

| Dimension | Weight | Signal |
|-----------|--------|--------|
| `semantic_similarity` | 30% | BGE embedding cosine similarity |
| `skill_overlap` | 25% | Required skill coverage |
| `experience_match` | 15% | Candidate years vs. job requirement |
| `education_match` | 5% | Degree relevance (tiered: CS/AI → Math → other) |
| `title_similarity` | 10% | Current title vs. job title (SequenceMatcher) |
| `location_match` | 5% | Geographic affinity |
| `behavioral_signal_score` | 10% | Open-to-work flag, GitHub activity, leadership score |

### Fallback Behaviour

| Dependency | Missing → Fallback |
|---|---|
| `sentence-transformers` / model weights | Lexical token overlap + BM25 |
| `faiss-cpu` | NumPy cosine similarity matrix |
| Cross-encoder reranker | RRF over feature and lexical ranks |

---

## Quick Start

### Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -r requirements.txt

# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

### Run

```bash
# Verify configuration
python main.py health

# Generate submission.csv
python main.py rank

# Validate submission format
python tools/validate_submission.py submission.csv

# Launch Streamlit dashboard
streamlit run app.py

# Start FastAPI REST server
uvicorn src.api:app --reload
```

---

## Data Files

Place challenge inputs in the `data/` directory:

| File | Purpose |
|------|---------|
| `sample_candidates.json` | Candidate profiles (JSON array or `candidates` key) |
| `candidate_schema.json` | JSON schema for candidate fields |
| `sample_submission.csv` | Template for ranked output format |
| `job_description.docx` | Target job posting (DOCX or JSONL) |

---

## REST API

When the FastAPI server is running (`uvicorn src.api:app`):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/datasets` | Summarise local candidate and job datasets |
| `POST` | `/rank` | Rank candidates (body: `RankRequest`) |
| `GET` | `/rank/{job_id}` | Rank candidates (convenience GET with query params) |
| `POST` | `/explain/{candidate_id}` | Generate structured recruiter explanation |

Interactive Swagger docs: `http://localhost:8000/docs`

### Request Example

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_description", "top_k": 10, "page": 1, "page_size": 5}'
```

### Pagination

`RankRequest` supports `page` (1-based) and `page_size` (1–100). `RankResponse` includes `total` count for client-side pagination logic.

---

## Environment Variables

All settings can be overridden via environment variables with the `RECRUITAI_` prefix, or placed in a `.env` file at the project root:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECRUITAI_PROJECT_NAME` | `RecruitAI` | Project display name |
| `RECRUITAI_DATA_DIR` | `data` | Path to dataset directory |
| `RECRUITAI_OUTPUTS_DIR` | `outputs` | Path for generated outputs |
| `RECRUITAI_LOGS_DIR` | `logs` | Path for log files |
| `RECRUITAI_MODELS_DIR` | `models` | Path for cached model weights |
| `RECRUITAI_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RECRUITAI_FAISS_INDEX_NAME` | `candidate_index.faiss` | FAISS index filename |
| `RECRUITAI_API_HOST` | `0.0.0.0` | FastAPI bind host |
| `RECRUITAI_API_PORT` | `8000` | FastAPI bind port |

---

## Testing

```bash
# Run all 129 tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**129 tests across 13 modules — all pass offline** without model downloads or network access.

| Test Module | Tests | Covers |
|-------------|-------|--------|
| `test_api.py` | 9 | Health, dataset summary, rank, error cases, parametrized top_k |
| `test_architecture.py` | 3 | Settings, engine construction, retrieval integration |
| `test_dashboard.py` | 8 | Pure metrics, mocked load, parametrized scores, edge cases |
| `test_data_loader.py` | 12 | JSON/JSONL/CSV/DOCX loading, flattening, schema inference |
| `test_embedding.py` | 8 | Cache, `coerce_documents`, cache-key stability |
| `test_explainer.py` | 15 | Single explain, all features, percentile context, batch, edge cases |
| `test_feature_engineering.py` | 20 | Column names, BM25, RRF, tokenizer, score bounds, parametrized |
| `test_preprocessing.py` | 8 | Text normalization, skill normalization, nested + flat profiles |
| `test_ranking_engine.py` | 13 | Construction, empty/single input, score bounds, ranks, fingerprint |
| `test_reranker.py` | 4 | Cross-encoder (normalized), fallback sort, lexical score |
| `test_scoring.py` | 16 | Weight validation, breakdown, NaN handling, clips, backward compat |
| `test_skill_extractor.py` | 11 | Extraction, fuzzy match, deduplication, parametrized skills |
| `test_vector_store.py` | 2 | Retrieval, save/load round-trip |

---

## Dashboard

```bash
streamlit run app.py
```

The **4-tab professional Streamlit dashboard** includes:

| Tab | Contents |
|-----|----------|
| 📈 **Overview** | KPI row (count, avg/top score, strong-fit %), top-N bar chart with tier colouring, score histogram |
| 👤 **Candidates** | Filterable ranked table, candidate detail panel with score progress bars and radar chart, CSV export |
| 🔬 **Analysis** | Score component heatmap (RdYlGn) across top-20 candidates |
| 📖 **About** | Pipeline diagram, score weight table, CLI reference |

---

## Project Layout

```
main.py                   CLI entry point  (python main.py rank)
app.py                    Streamlit entry  (streamlit run app.py)
config.py                 Top-level config re-export
requirements.txt          Full dependency list
pytest.ini                Test configuration (basetemp, testpaths)

src/
  config.py               Settings — pydantic-settings, env-variable overrides
  data_loader.py          JSON, JSONL, CSV, DOCX ingestion + schema validation
  preprocessing.py        Text cleaning, skill normalization, semantic profiles
  embedding.py            BGE embedding wrapper with disk cache
  vector_store.py         FAISS / NumPy vector index — is_built, add_vectors()
  feature_engineering.py  Multi-signal features: semantic + lexical + BM25 + RRF
  scoring.py              Weighted scoring — validate(), score_breakdown(), as_dict()
  reranker.py             Cross-encoder / normalize_scores() / RRF fallback
  ranking_engine.py       Hybrid pipeline orchestration + RankingMetrics
  explainer.py            Batch-calibrated recruiter explanations — explain_batch()
  api.py                  FastAPI — rank, explain, dataset endpoints, pagination
  dashboard.py            Streamlit — 4-tab professional UI
  skill_extractor.py      spaCy + rapidfuzz skill extraction
  logging_utils.py        Loguru / stdlib logging adapter
  utils.py                Directory creation and logging helpers

tests/
  conftest.py             Shared fixtures, fake ML components (offline)
  test_architecture.py    Smoke tests
  test_api.py             API unit tests (mocked dependencies)
  test_dashboard.py       Dashboard data helpers (mocked)
  test_data_loader.py     Data ingestion
  test_embedding.py       Embedding cache
  test_explainer.py       RankingExplainer with edge cases
  test_feature_engineering.py  Feature extraction and BM25/RRF
  test_preprocessing.py   Text normalization and profiles
  test_ranking_engine.py  End-to-end with fake dependencies
  test_reranker.py        Reranker strategies
  test_scoring.py         Weighted scoring and breakdown
  test_skill_extractor.py SkillExtractor
  test_vector_store.py    FAISS / NumPy retrieval

tools/
  validate_submission.py  Submission CSV format validator

data/                     Challenge dataset directory
outputs/                  Generated CSVs and FAISS indexes
models/                   Cached model weights (sentence-transformers)
logs/                     Application log files
.tmp/                     pytest temporary directory (auto-managed)
```

---

## Models

| Component | Model | Size |
|-----------|-------|------|
| Embedding | `BAAI/bge-small-en-v1.5` | 33M params, 512-dim |
| Reranker | `BAAI/bge-reranker-base` | optional, cross-encoder |

Both are downloaded from HuggingFace Hub on first run and cached in `models/sentence_transformers/`. All subsequent runs are fully **offline**.

---

## License

MIT — see [LICENSE](LICENSE).
