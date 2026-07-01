"""Text processing and lightweight information retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\+\#\.-]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in TOKEN_PATTERN.findall(normalized) if token not in STOP_WORDS]


def skill_key(skill: str) -> str:
    return normalize_text(skill).replace("-", " ")


@dataclass
class SearchHit:
    document_id: str
    score: float


class TfidfIndex:
    """A compact sparse TF-IDF index suitable for interview-scale demos."""

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        self.doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in self.doc_tokens.items()}
        self.average_doc_length = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        self.document_frequency: dict[str, int] = defaultdict(int)
        self.term_frequencies: dict[str, Counter[str]] = {}
        for doc_id, tokens in self.doc_tokens.items():
            counts = Counter(tokens)
            self.term_frequencies[doc_id] = counts
            for token in counts:
                self.document_frequency[token] += 1
        self.document_count = len(self.documents)

    def _idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        return math.log((1 + self.document_count) / (1 + df)) + 1.0

    def tfidf_vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        if not counts:
            return {}
        max_tf = max(counts.values())
        vector: dict[str, float] = {}
        for term, count in counts.items():
            tf = 0.5 + 0.5 * (count / max_tf)
            vector[term] = tf * self._idf(term)
        return vector

    @staticmethod
    def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        overlap = set(left).intersection(right)
        numerator = sum(left[term] * right[term] for term in overlap)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def semantic_scores(self, query: str) -> dict[str, float]:
        query_vector = self.tfidf_vector(query)
        scores: dict[str, float] = {}
        for doc_id, tokens in self.doc_tokens.items():
            doc_text = " ".join(tokens)
            doc_vector = self.tfidf_vector(doc_text)
            scores[doc_id] = self.cosine_similarity(query_vector, doc_vector)
        return scores

    def bm25_scores(self, query: str, k1: float, b: float) -> dict[str, float]:
        query_terms = tokenize(query)
        if not query_terms:
            return {doc_id: 0.0 for doc_id in self.documents}
        scores: dict[str, float] = {}
        for doc_id, term_counts in self.term_frequencies.items():
            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((self.document_count - self.document_frequency.get(term, 0) + 0.5) / (self.document_frequency.get(term, 0) + 0.5) + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / max(self.average_doc_length, 1)))
                score += idf * ((tf * (k1 + 1)) / denominator)
            scores[doc_id] = score
        return scores


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        ordered = sorted(ranking, key=lambda hit: hit.score, reverse=True)
        for rank, hit in enumerate(ordered, start=1):
            fused[hit.document_id] += 1.0 / (k + rank)
    return dict(fused)
