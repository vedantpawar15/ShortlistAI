"""Configuration primitives for RecruitAI."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureWeights:
    semantic_similarity: float = 0.28
    bm25_relevance: float = 0.12
    required_skill_overlap: float = 0.20
    preferred_skill_overlap: float = 0.08
    experience_fit: float = 0.12
    title_alignment: float = 0.07
    location_alignment: float = 0.05
    recency: float = 0.03
    achievement_intensity: float = 0.05

    def normalized(self) -> "FeatureWeights":
        total = sum(self.__dict__.values())
        return FeatureWeights(**{name: value / total for name, value in self.__dict__.items()})


@dataclass(frozen=True)
class ExplainabilityConfig:
    top_skill_evidence: int = 5
    max_reason_lines: int = 6


@dataclass(frozen=True)
class SearchConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    semantic_floor: float = 0.02


@dataclass(frozen=True)
class ScoringConfig:
    feature_weights: FeatureWeights = field(default_factory=lambda: FeatureWeights().normalized())
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


DEFAULT_SCORING_CONFIG = ScoringConfig()
