"""Skill extraction interfaces using lexical and NLP methods."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz
except ImportError:
    class fuzz:
        """Fallback subset of rapidfuzz.fuzz."""

        @staticmethod
        def ratio(left: object, right: object) -> float:
            return SequenceMatcher(None, str(left), str(right)).ratio() * 100

try:
    import spacy
except ImportError:
    spacy = None

from src.preprocessing import DEFAULT_SKILL_ALIASES, SkillNormalizer


class SkillExtractor:
    """Extract and normalize skills from resumes and job descriptions."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self.nlp: Any | None = None
        self.normalizer = SkillNormalizer()

    def load(self) -> None:
        """Load the configured spaCy pipeline."""
        if self.nlp is not None:
            return
        if spacy is None:
            return
        try:
            self.nlp = spacy.load(self.spacy_model)
        except OSError:
            self.nlp = spacy.blank("en")

    def extract(self, text: str) -> list[str]:
        """Extract candidate skills from text."""
        self.load()
        lowered = text.lower()
        candidates = [
            canonical
            for alias, canonical in DEFAULT_SKILL_ALIASES.items()
            if re.search(rf"\b{re.escape(alias)}\b", lowered)
        ]
        if self.nlp is not None:
            document = self.nlp(text)
            try:
                candidates.extend(chunk.text for chunk in document.noun_chunks if len(chunk.text.split()) <= 4)
            except ValueError:
                pass
        return self.normalizer.normalize_many(candidates)

    def fuzzy_match(self, skill: str, vocabulary: list[str]) -> tuple[str | None, float]:
        """Return the best fuzzy skill match from a controlled vocabulary."""
        if not vocabulary:
            return None, 0.0
        best_skill = max(vocabulary, key=lambda item: fuzz.ratio(skill, item))
        return best_skill, float(fuzz.ratio(skill, best_skill))

