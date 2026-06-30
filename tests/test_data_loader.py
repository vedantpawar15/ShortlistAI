"""Tests for local dataset ingestion."""

from __future__ import annotations

import json
import zipfile

import pandas as pd

from src.data_loader import (
    Candidate,
    DataLoader,
    Job,
    Submission,
    flatten_record,
    normalize_field_name,
    read_docx,
)


def test_pydantic_models_accept_extra_fields() -> None:
    candidate = Candidate(candidate_id="c1", name="Asha", custom_signal=0.91)
    job = Job(job_id="j1", title="ML Engineer", business_unit="AI")
    submission = Submission(candidate_id="c1", rank=1, score=0.98, reviewer_note="strong")

    assert candidate.candidate_id == "c1"
    assert candidate.model_extra == {"custom_signal": 0.91}
    assert job.job_id == "j1"
    assert job.model_extra == {"business_unit": "AI"}
    assert submission.score == 0.98
    assert submission.model_extra == {"reviewer_note": "strong"}


def test_normalize_field_name_handles_spaces_symbols_and_camel_case() -> None:
    assert normalize_field_name("Candidate ID") == "candidate_id"
    assert normalize_field_name("resumeText") == "resume_text"
    assert normalize_field_name("Contact.Email") == "contact_email"


def test_flatten_record_flattens_nested_dicts_and_lists() -> None:
    record = {
        "Candidate ID": "c1",
        "Contact": {"Email": "a@example.com"},
        "Skills": ["Python", "NLP"],
        "Experience": [{"Company": "Acme", "Years": 2}, {"Company": "Beta", "Years": 3}],
        "Portfolio": [],
    }

    flattened = flatten_record(record)

    assert flattened["candidate_id"] == "c1"
    assert flattened["contact_email"] == "a@example.com"
    assert flattened["skills"] == "Python | NLP"
    assert flattened["experience_0_company"] == "Acme"
    assert flattened["experience_1_years"] == 3
    assert flattened["portfolio"] is None


def test_load_candidates_auto_detects_json_and_normalizes_records(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "candidates": [
            {
                "Candidate ID": "c1",
                "Name": "Asha Rao",
                "Contact": {"Email": "asha@example.com"},
                "Skills": ["Python", "FAISS"],
                "Missing Field": "N/A",
            }
        ]
    }
    (data_dir / "candidate_json.json").write_text(json.dumps(payload), encoding="utf-8")

    frame = DataLoader(data_dir).load_candidates()

    assert list(frame["candidate_id"]) == ["c1"]
    assert list(frame["contact_email"]) == ["asha@example.com"]
    assert list(frame["skills"]) == ["Python | FAISS"]
    assert pd.isna(frame.loc[0, "missing_field"])


def test_load_jobs_supports_jsonl(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"jobId": "j1", "Title": "AI Engineer", "Required Skills": ["Python", "ML"]},
        {"jobId": "j2", "Title": "Data Scientist", "Required Skills": ["SQL"]},
    ]
    (data_dir / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    frame = DataLoader(data_dir).load_jobs()

    assert frame.shape[0] == 2
    assert list(frame["job_id"]) == ["j1", "j2"]
    assert list(frame["required_skills"]) == ["Python | ML", "SQL"]


def test_load_submission_supports_csv_and_schema_inference(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text(
        "Job ID,Candidate ID,Score\nj1,c1,0.95\nj1,c2,\n",
        encoding="utf-8",
    )

    loader = DataLoader(data_dir)
    frame = loader.load_submission()
    schema = loader.infer_schema(frame)

    assert list(frame.columns) == ["job_id", "candidate_id", "score"]
    assert frame.loc[0, "score"] == 0.95
    assert pd.isna(frame.loc[1, "score"])
    assert schema["job_id"] == "object"


def test_explicit_file_name_overrides_detection(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "unknown.csv").write_text("Candidate ID,Name\nc9,Dev\n", encoding="utf-8")

    frame = DataLoader(data_dir).load_candidates("unknown.csv")

    assert frame.loc[0, "candidate_id"] == "c9"
    assert frame.loc[0, "name"] == "Dev"


def test_load_candidate_schema_supports_standard_file(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    schema = {
        "title": "Candidate",
        "properties": {
            "candidate_id": {"type": "string"},
            "profile": {"properties": {"summary": {"type": "string"}}},
        },
    }
    (data_dir / "candidate_schema.json").write_text(json.dumps(schema), encoding="utf-8")

    frame = DataLoader(data_dir).load_candidate_schema()

    assert frame.loc[0, "title"] == "Candidate"
    assert frame.loc[0, "properties_candidate_id_type"] == "string"
    assert frame.loc[0, "properties_profile_properties_summary_type"] == "string"


def test_load_all_datasets_discovers_supported_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_candidates.json").write_text(
        json.dumps([{"candidate_id": "c1", "Name": "Asha"}]),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text(
        "candidate_id,rank,score\nc1,1,0.9\n",
        encoding="utf-8",
    )
    (data_dir / "candidates.jsonl").write_text(json.dumps({"candidate_id": "c2"}), encoding="utf-8")

    datasets = DataLoader(data_dir).load_all_datasets()

    assert set(datasets) == {"sample_candidates", "sample_submission", "candidates"}
    assert list(datasets["sample_candidates"]["candidate_id"]) == ["c1"]
    assert list(datasets["sample_submission"]["rank"]) == [1]
    assert list(datasets["candidates"]["candidate_id"]) == ["c2"]


def test_load_standard_files_includes_optional_challenge_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_candidates.json").write_text(json.dumps([{"candidate_id": "c1"}]), encoding="utf-8")
    (data_dir / "candidate_schema.json").write_text(json.dumps({"title": "Candidate"}), encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text(
        "candidate_id,rank,score\nc1,1,0.9\n",
        encoding="utf-8",
    )

    loaded = DataLoader(data_dir).load_standard_files()

    assert set(loaded) == {"sample_candidates", "candidate_schema", "sample_submission"}
    assert all(isinstance(frame, pd.DataFrame) for frame in loaded.values())


def test_read_docx_extracts_job_description_text(tmp_path) -> None:
    docx_path = tmp_path / "job_description.docx"
    xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Senior AI Engineer</w:t></w:r></w:p>
        <w:p><w:r><w:t>Build offline ranking systems.</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/document.xml", xml)

    records = read_docx(docx_path)

    assert records[0]["job_id"] == "job_description"
    assert records[0]["title"] == "Job Description"
    assert "Senior AI Engineer" in records[0]["description"]
    assert "Build offline ranking systems." in records[0]["description"]


def test_load_jobs_auto_detects_docx(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    docx_path = data_dir / "job_description.docx"
    xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Senior AI Engineer</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/document.xml", xml)

    frame = DataLoader(data_dir).load_jobs()

    assert frame.loc[0, "job_id"] == "job_description"
    assert "Senior AI Engineer" in frame.loc[0, "description"]


