"""Small WSGI API for ranking requests."""

from __future__ import annotations

import json
from http import HTTPStatus
from wsgiref.util import setup_testing_defaults

from .data.sample_data import sample_candidates, sample_job
from .pipeline import RecruitAIRanker


class RecruitAIAPI:
    def __init__(self) -> None:
        self.ranker = RecruitAIRanker()

    def __call__(self, environ, start_response):
        setup_testing_defaults(environ)
        method = environ["REQUEST_METHOD"]
        path = environ.get("PATH_INFO", "/")
        try:
            if method == "GET" and path == "/health":
                return self._respond(start_response, HTTPStatus.OK, {"status": "ok"})
            if method == "GET" and path == "/sample":
                return self._respond(
                    start_response,
                    HTTPStatus.OK,
                    {
                        "job": sample_job().to_dict(),
                        "candidates": [candidate.to_dict() for candidate in sample_candidates()],
                    },
                )
            if method == "POST" and path == "/rank":
                content_length = int(environ.get("CONTENT_LENGTH", "0") or "0")
                raw_body = environ["wsgi.input"].read(content_length).decode("utf-8")
                payload = json.loads(raw_body or "{}")
                results = self.ranker.rank_from_dicts(
                    job_payload=payload["job"],
                    candidate_payloads=payload["candidates"],
                )
                return self._respond(start_response, HTTPStatus.OK, {"results": results})
            return self._respond(start_response, HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except KeyError as exc:
            return self._respond(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": f"missing field: {exc.args[0]}"},
            )
        except json.JSONDecodeError:
            return self._respond(start_response, HTTPStatus.BAD_REQUEST, {"error": "invalid json"})

    @staticmethod
    def _respond(start_response, status: HTTPStatus, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        start_response(
            f"{status.value} {status.phrase}",
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]
