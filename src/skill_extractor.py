"""Skill extraction interfaces using lexical and NLP methods."""

from __future__ import annotations

from rapidfuzz import fuzz
import spacy


class SkillExtractor:
    """Extract and normalize skills from resumes and job descriptions."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self.nlp: spacy.language.Language | None = None

    def load(self) -> None:
        """Load the configured spaCy pipeline."""
        raise NotImplementedError("spaCy loading will be implemented later.")

    def extract(self, text: str) -> list[str]:
        """Extract candidate skills from text."""
        raise NotImplementedError("Skill extraction will be implemented later.")

    def fuzzy_match(self, skill: str, vocabulary: list[str]) -> tuple[str | None, float]:
        """Return the best fuzzy skill match from a controlled vocabulary."""
        if not vocabulary:
            return None, 0.0
        best_skill = max(vocabulary, key=lambda item: fuzz.ratio(skill, item))
        return best_skill, float(fuzz.ratio(skill, best_skill))

