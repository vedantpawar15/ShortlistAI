"""Sample entities for local demos and tests."""

from __future__ import annotations

from ..domain import CandidateProfile, JobRequirement


def sample_job() -> JobRequirement:
    return JobRequirement(
        job_id="job-ml-001",
        title="Senior Applied AI Engineer",
        summary=(
            "Build retrieval-augmented ranking systems, production LLM services, and explainable "
            "candidate matching pipelines for enterprise recruiting workflows."
        ),
        required_skills=[
            "Python",
            "Machine Learning",
            "NLP",
            "Information Retrieval",
            "APIs",
            "Testing",
        ],
        preferred_skills=[
            "Streamlit",
            "MLOps",
            "Prompt Engineering",
            "Ranking",
            "Explainability",
        ],
        minimum_years_experience=5,
        location="Remote - India",
        responsibilities=[
            "Design hybrid retrieval and scoring models",
            "Ship production APIs and dashboards",
            "Create evaluation harnesses and unit tests",
        ],
        keywords=["semantic search", "hybrid scoring", "recruiting", "LLM systems"],
    )


def sample_candidates() -> list[CandidateProfile]:
    return [
        CandidateProfile(
            candidate_id="cand-001",
            name="Aarav Sharma",
            headline="Senior AI Engineer | Search, Ranking, and NLP Systems",
            summary=(
                "Built semantic retrieval and ranking services for talent intelligence products using "
                "Python, NLP, evaluation pipelines, and production APIs."
            ),
            skills=[
                "Python",
                "Machine Learning",
                "NLP",
                "Information Retrieval",
                "APIs",
                "Testing",
                "Streamlit",
                "Ranking",
                "Explainability",
            ],
            years_experience=7,
            location="Remote - India",
            desired_titles=["Staff AI Engineer", "Senior Applied AI Engineer"],
            work_experiences=[
                "Led ranking architecture for an enterprise hiring marketplace",
                "Built search relevance features and evaluation tooling",
            ],
            achievements=[
                "Improved recruiter shortlist precision by 21% in A/B tests",
                "Reduced ranking latency from 380ms to 140ms",
            ],
            education=["B.Tech Computer Science"],
            certifications=["AWS Machine Learning Specialty"],
            updated_days_ago=5,
        ),
        CandidateProfile(
            candidate_id="cand-002",
            name="Maya Patel",
            headline="ML Engineer | Forecasting and Analytics",
            summary=(
                "Experienced in classical ML, dashboards, and backend APIs, with limited information "
                "retrieval and ranking exposure."
            ),
            skills=[
                "Python",
                "Machine Learning",
                "APIs",
                "SQL",
                "Testing",
                "Streamlit",
            ],
            years_experience=6,
            location="Bengaluru, India",
            desired_titles=["Senior Machine Learning Engineer"],
            work_experiences=[
                "Built churn prediction systems and monitoring pipelines",
                "Owned model packaging and service deployment",
            ],
            achievements=[
                "Raised demand forecast accuracy by 13%",
                "Cut incident volume by 30% with test automation",
            ],
            education=["M.Tech Data Science"],
            certifications=[],
            updated_days_ago=20,
        ),
        CandidateProfile(
            candidate_id="cand-003",
            name="Noah Kim",
            headline="Platform Engineer | Distributed Systems",
            summary=(
                "Strong backend engineer with service design and performance experience, but minimal "
                "NLP or recruiting-domain specialization."
            ),
            skills=["Python", "APIs", "Kubernetes", "Distributed Systems", "Testing", "MLOps"],
            years_experience=8,
            location="Remote - Singapore",
            desired_titles=["Senior Backend Engineer"],
            work_experiences=[
                "Scaled document processing services to 50M requests per day",
                "Improved API reliability and deployment safety",
            ],
            achievements=[
                "Delivered 99.97% availability for core screening APIs",
                "Reduced cloud cost by 18% through autoscaling controls",
            ],
            education=["B.S. Software Engineering"],
            certifications=["CKA"],
            updated_days_ago=8,
        ),
    ]
