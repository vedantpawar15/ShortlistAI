from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recruit_ai.api import RecruitAIAPI
from recruit_ai.data.sample_data import sample_candidates, sample_job


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = RecruitAIAPI()

    def _call(self, method: str, path: str, body: dict | None = None) -> tuple[str, dict]:
        raw = json.dumps(body).encode("utf-8") if body is not None else b""
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "CONTENT_LENGTH": str(len(raw)),
            "wsgi.input": io.BytesIO(raw),
        }
        captured = {}

        def start_response(status, headers):
            captured["status"] = status
            captured["headers"] = headers

        response = b"".join(self.app(environ, start_response))
        return captured["status"], json.loads(response.decode("utf-8"))

    def test_health_endpoint(self) -> None:
        status, payload = self._call("GET", "/health")
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["status"], "ok")

    def test_rank_endpoint(self) -> None:
        job = sample_job().to_dict()
        candidates = [candidate.to_dict() for candidate in sample_candidates()]
        status, payload = self._call("POST", "/rank", {"job": job, "candidates": candidates})
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["results"][0]["candidate"]["candidate_id"], "cand-001")


if __name__ == "__main__":
    unittest.main()
