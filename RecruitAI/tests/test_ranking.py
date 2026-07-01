from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recruit_ai.data.sample_data import sample_candidates, sample_job
from recruit_ai.pipeline import RecruitAIRanker


class RankingTests(unittest.TestCase):
    def test_best_candidate_has_full_required_coverage(self) -> None:
        results = RecruitAIRanker().rank(sample_job(), sample_candidates())
        self.assertEqual(results[0].candidate.candidate_id, "cand-001")
        self.assertEqual(results[0].missing_required_skills, [])

    def test_missing_required_skills_penalize_score(self) -> None:
        results = RecruitAIRanker().rank(sample_job(), sample_candidates())
        self.assertGreater(results[0].total_score, results[1].total_score)
        self.assertGreater(results[1].total_score, results[2].total_score)

    def test_explanations_include_weighted_drivers(self) -> None:
        results = RecruitAIRanker().rank(sample_job(), sample_candidates())
        explanation_text = " ".join(results[0].explanation)
        self.assertIn("Top weighted drivers", explanation_text)
        self.assertIn("required-skill", explanation_text)


if __name__ == "__main__":
    unittest.main()
