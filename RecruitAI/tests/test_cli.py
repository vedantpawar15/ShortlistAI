from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main


class CLITests(unittest.TestCase):
    def test_demo_json_output(self) -> None:
        from io import StringIO

        buffer = StringIO()
        previous_stdout = sys.stdout
        try:
            sys.stdout = buffer
            exit_code = main.run_demo(as_json=True)
        finally:
            sys.stdout = previous_stdout
        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload[0]["candidate"]["candidate_id"], "cand-001")


if __name__ == "__main__":
    unittest.main()
