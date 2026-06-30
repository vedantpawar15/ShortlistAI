"""Data ingestion utilities for local RecruitAI datasets."""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas.api import types as pd_types
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.logging_utils import logger

DatasetKind = Literal["candidates", "jobs", "submission", "candidate_schema"]

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson", ".docx"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "nil", "missing"}
STANDARD_DATASET_FILES: dict[str, DatasetKind] = {
    "sample_candidates.json": "candidates",
    "candidate_schema.json": "candidate_schema",
    "sample_submission.csv": "submission",
    "job_description.docx": "jobs",
}


class Candidate(BaseModel):
    """Normalized candidate record with permissive extra fields."""

    candidate_id: str | None = None
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    resume_text: str | None = None
    skills: str | list[str] | None = None
    experience_years: float | None = None
    education: str | None = None
    location: str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator(
        "candidate_id",
        "name",
        "email",
        "phone",
        "resume_text",
        "education",
        "location",
        mode="before",
    )
    @classmethod
    def _empty_text_to_none(cls, value: Any) -> Any:
        return none_if_missing(value)


class Job(BaseModel):
    """Normalized job description record with permissive extra fields."""

    job_id: str | None = None
    title: str | None = None
    description: str | None = None
    required_skills: str | list[str] | None = None
    preferred_skills: str | list[str] | None = None
    experience_years: float | None = None
    education: str | None = None
    location: str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("job_id", "title", "description", "education", "location", mode="before")
    @classmethod
    def _empty_text_to_none(cls, value: Any) -> Any:
        return none_if_missing(value)


class Submission(BaseModel):
    """Normalized ranked submission record with permissive extra fields."""

    candidate_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    score: float | None = Field(default=None, ge=0)
    reasoning: str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("candidate_id", "job_id", "reasoning", mode="before")
    @classmethod
    def _empty_text_to_none(cls, value: Any) -> Any:
        return none_if_missing(value)


class DataLoader:
    """Load local candidate, job, submission, and schema datasets from disk."""

    detection_keywords: dict[DatasetKind, tuple[str, ...]] = {
        "candidates": ("candidate", "candidates", "resume", "resumes", "applicant", "applicants"),
        "jobs": ("job", "jobs", "description", "descriptions", "posting", "postings"),
        "submission": ("submission", "sample_submission", "sample-submission", "sample"),
        "candidate_schema": ("candidate_schema",),
    }

    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_candidates(self, file_name: str | Path | None = None) -> pd.DataFrame:
        """Load candidate records as a normalized DataFrame."""
        frame = self._load_dataset("candidates", file_name)
        return self._validate_records(frame, Candidate, "candidate")

    def load_jobs(self, file_name: str | Path | None = None) -> pd.DataFrame:
        """Load job description records as a normalized DataFrame."""
        frame = self._load_dataset("jobs", file_name)
        return self._validate_records(frame, Job, "job")

    def load_submission(self, file_name: str | Path | None = None) -> pd.DataFrame:
        """Load sample submission records as a normalized DataFrame."""
        frame = self._load_dataset("submission", file_name)
        return self._validate_records(frame, Submission, "submission")

    def load_candidate_schema(self, file_name: str | Path | None = None) -> pd.DataFrame:
        """Load the candidate JSON schema as a flattened DataFrame when available."""
        return self._load_dataset("candidate_schema", file_name)

    def load_standard_files(self) -> dict[str, pd.DataFrame]:
        """Load known challenge files from the data directory if they exist."""
        loaded: dict[str, pd.DataFrame] = {}
        for file_name, kind in STANDARD_DATASET_FILES.items():
            path = self.data_dir / file_name
            if not path.exists():
                logger.info("Optional standard dataset file is not present: {}", path)
                continue
            loaded[path.stem] = self._load_dataset(kind, path)
        return loaded

    def load_all_datasets(self) -> dict[str, pd.DataFrame]:
        """Automatically load every supported dataset file under the data directory."""
        datasets: dict[str, pd.DataFrame] = {}
        for path in self._supported_files():
            kind = self._infer_kind_from_path(path) or "candidate_schema"
            try:
                datasets[path.stem] = self._load_dataset(kind, path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                logger.error("Failed to load dataset file {}: {}", path, exc)
                raise
        return datasets

    def discover_files(self) -> dict[DatasetKind, list[Path]]:
        """Return supported dataset files grouped by inferred dataset kind."""
        candidates = self._supported_files()
        discovered: dict[DatasetKind, list[Path]] = {
            "candidates": [],
            "jobs": [],
            "submission": [],
            "candidate_schema": [],
        }
        for path in candidates:
            kind = self._infer_kind_from_path(path)
            if kind is not None:
                discovered[kind].append(path)
        logger.info(
            "Discovered dataset files: candidates={}, jobs={}, submission={}, candidate_schema={}",
            len(discovered["candidates"]),
            len(discovered["jobs"]),
            len(discovered["submission"]),
            len(discovered["candidate_schema"]),
        )
        return discovered

    def infer_schema(self, frame: pd.DataFrame) -> dict[str, str]:
        """Infer a simple column-to-dtype schema from a DataFrame."""
        return {column: normalized_dtype_name(dtype) for column, dtype in frame.dtypes.items()}

    def validate_schema(self, frame: pd.DataFrame, required_columns: list[str]) -> bool:
        """Validate that a dataset contains the expected columns."""
        normalized_required = {normalize_field_name(column) for column in required_columns}
        missing_columns = normalized_required - set(frame.columns)
        if missing_columns:
            logger.warning("Dataset is missing required columns: {}", sorted(missing_columns))
        return not missing_columns

    def _load_dataset(self, kind: DatasetKind, file_name: str | Path | None) -> pd.DataFrame:
        path = self._resolve_dataset_path(kind, file_name)
        logger.info("Loading {} dataset from {}", kind, path)
        try:
            records = self._read_records(path)
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
            ET.ParseError,
            pd.errors.ParserError,
        ) as exc:
            logger.error("Unable to read {} dataset from {}: {}", kind, path, exc)
            raise ValueError(f"Unable to read {kind} dataset from {path}") from exc
        flat_records = [flatten_record(record) for record in records]
        frame = pd.DataFrame(flat_records)
        frame = normalize_dataframe(frame)
        logger.info("Loaded {} {} records with {} columns", len(frame), kind, len(frame.columns))
        logger.debug("Inferred {} schema: {}", kind, self.infer_schema(frame))
        return frame

    def _resolve_dataset_path(self, kind: DatasetKind, file_name: str | Path | None) -> Path:
        if file_name is not None:
            path = Path(file_name)
            if not path.is_absolute() and not path.exists():
                path = self.data_dir / path
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported dataset file type: {path.suffix}")
            return path

        discovered = self.discover_files()[kind]
        if not discovered:
            raise FileNotFoundError(f"No {kind} dataset file found in {self.data_dir}")
        if len(discovered) > 1:
            names = ", ".join(path.name for path in discovered)
            logger.warning("Multiple {} files found; using {} from [{}]", kind, discovered[0].name, names)
        return discovered[0]

    def _supported_files(self) -> list[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        files = [
            path
            for path in self.data_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(files, key=dataset_sort_key)

    def _infer_kind_from_path(self, path: Path) -> DatasetKind | None:
        standard_kind = STANDARD_DATASET_FILES.get(path.name.lower())
        if standard_kind is not None:
            return standard_kind

        name = normalize_field_name(path.stem)
        for kind, keywords in self.detection_keywords.items():
            if any(normalize_field_name(keyword) in name for keyword in keywords):
                return kind
        return None

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path).to_dict(orient="records")
        if suffix in {".jsonl", ".ndjson"}:
            return read_jsonl(path)
        if suffix == ".json":
            return read_json(path)
        if suffix == ".docx":
            return read_docx(path)
        raise ValueError(f"Unsupported dataset file type: {suffix}")

    def _validate_records(self, frame: pd.DataFrame, model_type: type[BaseModel], label: str) -> pd.DataFrame:
        if frame.empty:
            logger.warning("Loaded {} dataset is empty", label)
            return frame

        for index, row in frame.iterrows():
            payload = row.dropna().to_dict()
            try:
                model_type.model_validate(payload)
            except Exception as exc:
                logger.warning("{} record at row {} did not validate cleanly: {}", label.capitalize(), index, exc)

        return frame


def read_json(path: Path) -> list[dict[str, Any]]:
    """Read JSON data and extract a list of records."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return coerce_records(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines data into record dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.extend(coerce_records(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def read_docx(path: Path) -> list[dict[str, Any]]:
    """Extract plain text from a DOCX file as a single dataset record."""
    document_xml = "word/document.xml"
    paragraphs: list[str] = []
    with zipfile.ZipFile(path) as archive:
        if document_xml not in archive.namelist():
            raise ValueError(f"DOCX file does not contain {document_xml}: {path}")
        with archive.open(document_xml) as document:
            root = ET.fromstring(document.read())

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        text_parts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    description = "\n".join(paragraphs).strip()
    return [
        {
            "job_id": normalize_field_name(path.stem),
            "title": path.stem.replace("_", " ").strip().title(),
            "description": description,
            "source_file": path.name,
        }
    ]


def coerce_records(payload: Any) -> list[dict[str, Any]]:
    """Convert common JSON payload shapes into a list of record dictionaries."""
    if isinstance(payload, list):
        return [coerce_record(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in ("candidates", "jobs", "submission", "data", "records", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [coerce_record(item) for item in value]
        return [dict(payload)]
    raise ValueError("Dataset payload must be a JSON object or list of objects")


def coerce_record(value: Any) -> dict[str, Any]:
    """Coerce one payload item into a dictionary record."""
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def flatten_record(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries and lists using normalized underscore keys."""
    flattened: dict[str, Any] = {}
    for raw_key, value in record.items():
        key = normalize_field_name(str(raw_key))
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_record(value, full_key))
        elif isinstance(value, list):
            flattened.update(flatten_list(full_key, value))
        else:
            flattened[full_key] = none_if_missing(value)
    return flattened


def flatten_list(key: str, values: list[Any]) -> dict[str, Any]:
    """Flatten list values while preserving useful scalar and nested content."""
    if not values:
        return {key: None}

    if all(not isinstance(item, (Mapping, list)) for item in values):
        cleaned = [str(item).strip() for item in values if none_if_missing(item) is not None]
        return {key: " | ".join(cleaned) if cleaned else None}

    flattened: dict[str, Any] = {key: json.dumps(values, ensure_ascii=True)}
    for index, item in enumerate(values):
        indexed_key = f"{key}_{index}"
        if isinstance(item, Mapping):
            flattened.update(flatten_record(item, indexed_key))
        elif isinstance(item, list):
            flattened.update(flatten_list(indexed_key, item))
        else:
            flattened[indexed_key] = none_if_missing(item)
    return flattened


def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, remove duplicate columns, and standardize missing values."""
    if frame.empty:
        return frame

    normalized = frame.copy()
    normalized.columns = deduplicate_columns([normalize_field_name(column) for column in normalized.columns])
    normalized = normalized.map(none_if_missing)
    normalized = normalized.where(pd.notna(normalized), None)
    return normalized


def dataset_sort_key(path: Path) -> tuple[int, str]:
    """Sort standard challenge files before other discovered files."""
    order = {name: index for index, name in enumerate(STANDARD_DATASET_FILES)}
    return (order.get(path.name.lower(), len(order)), path.name.lower())


def deduplicate_columns(columns: Iterable[str]) -> list[str]:
    """Make normalized column names unique while keeping them readable."""
    counts: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        base = column or "field"
        count = counts.get(base, 0)
        result.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return result


def normalize_field_name(value: object) -> str:
    """Normalize raw dataset field names to snake_case."""
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def none_if_missing(value: Any) -> Any:
    """Convert common missing-value representations to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return None
    if isinstance(value, (list, dict)) and not value:
        return None
    return value


def normalized_dtype_name(dtype: Any) -> str:
    """Return stable dtype names across pandas versions and string backends."""
    if pd_types.is_object_dtype(dtype) or pd_types.is_string_dtype(dtype):
        return "object"
    return str(dtype)
