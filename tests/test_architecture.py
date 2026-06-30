"""Architecture-level smoke tests."""

from src.config import Settings
from src.ranking_engine import RankingEngine


def test_settings_can_be_constructed() -> None:
    """Settings should be instantiable with defaults."""
    settings = Settings()
    assert settings.project_name == "RecruitAI"


def test_ranking_engine_can_be_constructed() -> None:
    """Ranking engine should expose the planned orchestration surface."""
    engine = RankingEngine()
    assert engine is not None

