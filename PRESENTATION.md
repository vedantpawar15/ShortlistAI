# ShortlistAI — Project Presentation Guide

> **Everything you need to explain this project confidently in a presentation, interview, or hackathon.**

---

## 1. What Is ShortlistAI?

ShortlistAI is an **AI-powered candidate ranking system** designed for recruiters.

Imagine a company posts a job for a *Machine Learning Engineer*. They receive **50+ resumes**. A recruiter has to manually read all of them and decide who to interview. That's slow, biased, and inconsistent.

**ShortlistAI automates this.** It reads all the candidate profiles and the job description, scores every candidate across 7 dimensions using AI, and returns a ranked list — best fit at the top — with a plain-English explanation for every ranking decision.

### In One Line
> *"Give me 50 resumes and a job description. I'll tell you who to call first — and why."*

---

## 2. The Problem It Solves

| Pain Point | Without ShortlistAI | With ShortlistAI |
|------------|--------------------|--------------------|
| Speed | Hours of manual reading | Ranked output in seconds |
| Consistency | Different recruiters rank differently | Deterministic, reproducible scores |
| Bias | Unconscious recency/name bias | Feature-based, explainable scoring |
| Scale | Painful at 50 CVs, impossible at 500 | Scales to thousands offline |
| Trust | "Why is this candidate #1?" | Per-candidate explanation with breakdown |

---

## 3. The Tech Stack

```
Language        Python 3.12
AI Models       BAAI/bge-small-en-v1.5  (embedding, HuggingFace)
                BAAI/bge-reranker-base  (reranking, optional)
Vector Search   FAISS  (Facebook AI Similarity Search)
Data            Pandas, NumPy
API             FastAPI + Uvicorn
Dashboard       Streamlit + Plotly
Validation      Pydantic v2
Testing         pytest  (129 tests, 100% offline)
Config          pydantic-settings (.env support)
Logging         Loguru
```

**Fully offline** — once model weights are downloaded once, the entire system runs without internet.

---

## 4. System Architecture — The Full Pipeline

This is the exact flow from raw files to ranked output:

```
  INPUT
  ┌─────────────────────────────┐
  │  sample_candidates.json     │  ← 50 candidate profiles
  │  job_description.docx       │  ← the job posting
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  1. DataLoader              │  Reads JSON, JSONL, CSV, DOCX
  │     ↳ field normalization   │  "Candidate ID" → "candidate_id"
  │     ↳ nested → flat schema  │  {"profile": {"name": ...}} → flat
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  2. TextPreprocessor        │  Cleans raw text
  │     ↳ unicode normalization │  "Café" → "cafe"
  │     ↳ punctuation removal   │  "Python, NLP!!!" → "python nlp"
  │     ↳ skill normalization   │  "JS" → "javascript", "k8s" → "kubernetes"
  │     ↳ semantic document     │  builds one rich text blob per candidate
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  3. EmbeddingModel          │  Neural text → numbers
  │     ↳ BGE-small-en-v1.5    │  512-dimensional vector per candidate
  │     ↳ disk cache            │  re-uses embeddings across runs
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  4. FaissVectorStore        │  Fast similarity search
  │     ↳ builds index once     │  cached by candidate fingerprint
  │     ↳ semantic top-K        │  finds 50 most similar candidates to job
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  5. FeatureEngineer         │  Builds 7 scored signals per candidate
  │     ↳ semantic_similarity   │  embedding cosine score (from FAISS)
  │     ↳ skill_overlap         │  Python, NLP ∩ required skills
  │     ↳ experience_match      │  candidate 6yr / required 5yr = 1.0
  │     ↳ education_match       │  CS degree → 1.0, Sales degree → 0.5
  │     ↳ title_similarity      │  "ML Eng" ≈ "ML Engineer" → 0.9
  │     ↳ location_match        │  "Pune" = "Pune" → 1.0
  │     ↳ behavioral_signals    │  GitHub score, open-to-work flag
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  6. CandidateScorer         │  Weighted sum → one composite score
  │     ↳ hybrid blend          │  60% semantic + 40% feature score
  │     ↳ score ∈ [0.0, 1.0]   │  always bounded
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  7. Reranker                │  Second-pass refinement
  │     ↳ cross-encoder (opt.)  │  BGE-reranker-base (highest quality)
  │     ↳ RRF fallback          │  Reciprocal Rank Fusion (deterministic)
  └─────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────┐
  │  8. RankingExplainer        │  "Why is this candidate #1?"
  │     ↳ batch percentiles     │  "top 8% of applicants"
  │     ↳ strengths + gaps      │  "Missing: Kubernetes"
  │     ↳ score breakdown       │  per-feature contribution
  └─────────────────────────────┘
              │
              ▼
  OUTPUT
  ┌─────────────────────────────┐
  │  submission.csv             │  candidate_id, rank, score, reasoning
  │  Streamlit Dashboard        │  visual charts + detail panels
  │  REST API (/rank, /explain) │  integrate into any recruiter tool
  └─────────────────────────────┘
```

---

## 5. Key AI Concepts Used

### 5.1 Embeddings (Semantic Understanding)
Instead of checking if the word "Python" appears in both the CV and job description, we convert both into **high-dimensional vectors**. Vectors that are close together in space are *semantically similar* — even if they use different words.

```
"ML engineer with Python experience"  →  [0.23, -0.41, 0.87, ...]  (512 numbers)
"machine learning developer, skilled in python"  →  [0.21, -0.39, 0.85, ...]

Cosine similarity = 0.97  ← very close → strong match
```

Model used: **BAAI/bge-small-en-v1.5** — one of the best small English embedding models (33M parameters, runs on CPU).

### 5.2 FAISS — Vector Search at Scale
Once all 50 candidate embeddings are computed, we store them in a **FAISS index** (Facebook AI Similarity Search). FAISS can find the top-K most similar candidates to the job embedding in milliseconds — even across millions of candidates.

### 5.3 BM25 — Lexical Relevance
A classic information retrieval formula that scores how relevant a document is based on **term frequency** and **document length normalisation**. It complements neural embeddings by catching exact keyword matches.

```
BM25("python kubernetes", candidate_doc) = 0.73  ← candidate mentions both terms
BM25("python kubernetes", another_doc)   = 0.12  ← candidate mentions neither
```

### 5.4 Reciprocal Rank Fusion (RRF)
When we have two ranking signals (e.g., FAISS retrieval order and feature-score order), we can't just average them — they live in different scales. **RRF** merges multiple ranked lists by rewarding candidates who rank high in *multiple* lists:

```
RRF_score(candidate) = Σ  1 / (k + rank_in_list_i)
                      for each list i
```

Where `k=60` is a smoothing constant. This is robust, parameter-free, and works better than simple score averaging.

### 5.5 Cross-Encoder Reranking (Optional)
After the initial scoring, we optionally run a **cross-encoder** that takes the *pair* `(job_description, candidate_text)` together and produces a relevance score. Cross-encoders are slower but more accurate because they can model interaction between the two texts.

Model: **BAAI/bge-reranker-base**

---

## 6. Scoring System — How Each Candidate Gets a Number

Every candidate gets a score between **0.0** (no fit) and **1.0** (perfect fit).

### The Formula

```
final_score = 0.60 × semantic_similarity
            + 0.40 × (
                  0.25 × skill_overlap
                + 0.15 × experience_match
                + 0.05 × education_match
                + 0.10 × title_similarity
                + 0.05 × location_match
                + 0.10 × behavioral_signal_score
              )
```

### Each Signal Explained

| Signal | How It's Calculated | Example |
|--------|---------------------|---------|
| **semantic_similarity** | Cosine similarity between BGE job embedding and candidate embedding | Job: "ML Engineer", Candidate "AI Developer" → 0.92 |
| **skill_overlap** | `(matched required skills) / (total required skills)` | Job needs Python, ML, NLP. Candidate has Python, ML → 0.67 |
| **experience_match** | `min(1.0, candidate_years / required_years)` | Required 5yr, candidate has 7yr → 1.0 |
| **education_match** | Tiered: CS/AI/Data → 1.0, Math/Physics → 0.75, Other → 0.5 | B.Tech Computer Science → 1.0 |
| **title_similarity** | SequenceMatcher ratio of current title vs. job title | "Senior ML Engineer" vs "ML Engineer" → 0.82 |
| **location_match** | Exact or substring city match | Both "Pune" → 1.0 |
| **behavioral_signal_score** | Aggregated from `open_to_work_flag`, `github_activity_score` | Active GitHub + open to work → 0.85 |

---

## 7. Explainability — Why Did This Candidate Rank #1?

Every ranking comes with a **human-readable explanation**:

```
Candidate: CAND_0042
Rank: #1
Score: 0.847

Summary: "Fit driven by overall score 0.847.
          This places the candidate in the top 8% of applicants."

Strengths:
  ✅ Strong semantic alignment with the job description
  ✅ Strong skill match: python, machine learning, natural language processing
  ✅ Experience level meets or exceeds the role requirement
  ✅ Located in or near the target location

Gaps:
  ⚠️ Missing required skills: kubernetes, docker

Score Breakdown:
  semantic_similarity    : 0.88
  skill_overlap          : 0.75
  experience_match       : 1.00
  education_match        : 1.00
  title_similarity       : 0.72
  location_match         : 1.00
  behavioral_signal_score: 0.80
```

Thresholds for "strength" vs "gap" are **dynamically calibrated** from the batch — using the 75th percentile as the "high" bar and 25th percentile as the "low" bar. This means the explanations always describe a candidate *relative to the current pool*, not hardcoded absolute values.

---

## 8. The Dashboard

Run `streamlit run app.py` to open the visual interface.

### Tab 1 — Overview
- **KPI row**: total candidates, average score, top score, % strong-fit candidates
- **Top-N bar chart**: horizontal bars coloured by tier (🟢 Strong / 🟡 Moderate / 🔴 Weak)
- **Score distribution histogram**: see how the applicant pool clusters

### Tab 2 — Candidates
- **Ranked table**: colour-gradient scoring, sortable
- **Candidate detail panel**: click any candidate to see:
  - Score progress bars per dimension
  - Radar/spider chart showing the 7-dimensional profile
  - Full reasoning text
- **CSV export button**: download the ranked list

### Tab 3 — Analysis
- **Score component heatmap**: see at a glance which candidates are strong/weak on each dimension

### Tab 4 — About
- Pipeline diagram, weight table, CLI reference

---

## 9. The REST API

Run `uvicorn src.api:app --reload` and use any HTTP client:

```
GET  /health              → { "status": "ok", "project": "RecruitAI" }
GET  /datasets            → { "candidates": 50, "jobs": ["job_description"] }
POST /rank                → ranked list with pagination
GET  /rank/job_description?top_k=10  → same, GET form
POST /explain/CAND_0042   → full CandidateExplanation JSON
```

Swagger UI auto-generated at `http://localhost:8000/docs`.

---

## 10. Fallback Design — Runs Everywhere

The system is designed to **gracefully degrade** when heavy dependencies are missing:

```
Full stack (GPU/CPU + internet on first run):
  FAISS + BGE embedding + cross-encoder reranker
  → best quality, fully neural pipeline

No sentence-transformers:
  BM25 term frequency + lexical token overlap
  → deterministic, reproducible, still reasonable quality

No FAISS:
  NumPy cosine similarity matrix
  → O(N) instead of O(log N), fine for < 10k candidates

No cross-encoder:
  Reciprocal Rank Fusion over feature-score and BM25 ranks
  → zero dependencies, mathematically sound
```

This means the project **always runs** — even in a constrained CI/CD environment with just `pandas`, `numpy`, and `pydantic` installed.

---

## 11. Test Suite

```bash
python -m pytest tests/ -v
# 129 passed in 47s — fully offline
```

- **13 test modules** covering every component
- All fast unit tests — fake ML models (no downloads)
- `conftest.py` provides shared fixtures and fake embedding/vector store classes
- Parametrized tests (`@pytest.mark.parametrize`) for edge cases
- `pytest.ini` configured with `--basetemp=.tmp` for cross-platform compatibility

---

## 12. How to Run a Full Demo

```bash
# Step 1 — Health check
python main.py health
# → RecruitAI is configured for project: RecruitAI

# Step 2 — Rank candidates (reads data/, writes submission.csv)
python main.py rank
# → Wrote 50 ranked candidates to submission.csv

# Step 3 — Validate output format
python tools/validate_submission.py submission.csv
# → Submission is valid

# Step 4 — Open the dashboard
streamlit run app.py
# → http://localhost:8501
```

---

## 13. Key Design Decisions (Good for Interview Questions)

### Q: Why FAISS + Feature scoring instead of just one approach?

> Neural embeddings are great at semantic understanding but they can miss structured signals like "candidate has exactly 5 years of experience" or "candidate is in Pune". Feature scoring captures those structured signals. We blend both with a configurable α — the best of both worlds.

### Q: Why RRF instead of score averaging?

> Score averaging requires both scores to be on the same scale, which they're not — cosine similarity lives in [-1, 1], feature scores in [0, 1], cross-encoder logits in (-∞, +∞). RRF is rank-based so it's scale-invariant, parameter-light, and consistently outperforms weighted score averaging in IR benchmarks (Cormack et al. 2009).

### Q: Why batch-calibrated thresholds in the explainer?

> Hardcoded thresholds (e.g., "score ≥ 0.8 is a strength") fail when the applicant pool is uniformly strong or uniformly weak. By deriving thresholds from the 75th/25th percentile of the batch, explanations always reflect a candidate's standing *within the actual pool* — which is what recruiters care about.

### Q: How does it handle missing data?

> Every feature has a safe default — missing skills → empty set (skill_overlap = 0), missing location → 0.0, missing experience → comparison returns 0.0. Pydantic v2 validators catch schema violations at load time. NaN values in the feature frame are filled with 0 before scoring.

### Q: Why is it offline-first?

> Recruiter workflows often happen in secure enterprise environments without internet access to external APIs. By using HuggingFace models cached locally and FAISS for vector search, the entire pipeline runs air-gapped after a one-time model download.

---

## 14. Metrics and Quality

After every ranking run, `RankingMetrics` logs:

```
RankingMetrics: total=50 retrieved=50 ranked=50
                top=0.9123 mean=0.6547 median=0.6412 std=0.1203
```

This gives a quick sanity check — if `mean` is too low, the job description may be too vague; if `std` is near 0, all candidates look identical to the model.

---

## 15. What Makes This Hackathon-Worthy

| Criterion | What We Did |
|-----------|-------------|
| **Novelty** | True hybrid scoring — not just embedding search, not just keyword match |
| **Technical depth** | BM25 + FAISS + RRF + cross-encoder + weighted features in one pipeline |
| **Explainability** | Every decision is interpretable by a non-technical recruiter |
| **Production-readiness** | FastAPI, Pydantic validation, Loguru logging, graceful fallbacks |
| **Testing discipline** | 129 tests, 0 failures, fake-dependency isolation |
| **UX** | Professional 4-tab Streamlit dashboard with radar charts and heatmaps |
| **Offline-first** | Works in air-gapped enterprise environments |
| **Extensibility** | Dependency injection throughout — swap any component with a custom one |
