"""Tests for SkillExtractor — lexical and NLP-backed skill extraction."""

from __future__ import annotations

import pytest

from src.skill_extractor import SkillExtractor


EXTRACTOR = SkillExtractor()


def test_extract_returns_list() -> None:
    result = EXTRACTOR.extract("Python and machine learning engineer")
    assert isinstance(result, list)


def test_extract_finds_alias_skills() -> None:
    result = EXTRACTOR.extract("Experienced in Python, NLP, and k8s.")
    assert "python" in result
    assert "natural language processing" in result
    assert "kubernetes" in result


def test_extract_handles_empty_string() -> None:
    result = EXTRACTOR.extract("")
    assert result == []


def test_extract_deduplicates_synonyms() -> None:
    # "ml" and "machine learning" are the same canonical skill.
    result = EXTRACTOR.extract("ml and machine learning specialist")
    assert result.count("machine learning") == 1


def test_fuzzy_match_finds_best_candidate() -> None:
    vocabulary = ["python", "java", "javascript", "machine learning"]
    best, score = EXTRACTOR.fuzzy_match("pythn", vocabulary)

    assert best == "python"
    assert score > 50  # SequenceMatcher ratio * 100


def test_fuzzy_match_returns_none_for_empty_vocabulary() -> None:
    best, score = EXTRACTOR.fuzzy_match("python", [])

    assert best is None
    assert score == 0.0


def test_fuzzy_match_works_with_exact_match() -> None:
    vocabulary = ["tensorflow", "pytorch", "scikit-learn"]
    best, score = EXTRACTOR.fuzzy_match("tensorflow", vocabulary)

    assert best == "tensorflow"
    assert score == 100.0


@pytest.mark.parametrize(
    "text,expected_skill",
    [
        ("We use Python for data pipelines.", "python"),
        ("Deep learning with TensorFlow.", "tensorflow"),
        ("Running jobs on k8s clusters.", "kubernetes"),
        ("JavaScript frontend development.", "javascript"),
    ],
)
def test_extract_parametrized_skill_detection(text: str, expected_skill: str) -> None:
    result = EXTRACTOR.extract(text)
    assert expected_skill in result
