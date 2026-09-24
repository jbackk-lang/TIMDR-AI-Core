import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online_learning import OnlineLearningError, run_cycle


class OnlineLearningTests(unittest.TestCase):
    def test_skips_one_unavailable_document_and_keeps_next_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "catalog.json"
            config.write_text(json.dumps({"items": [
                {"id": "bad", "url": "https://example.org/bad", "title": "bad"},
                {"id": "good", "url": "https://example.org/good", "title": "good"},
            ]}), encoding="utf-8")

            def fake_read(url, _limit):
                if url.endswith("bad"):
                    raise OnlineLearningError("not found")
                return b"modal vibration frequency"

            with patch("online_learning._read_url", side_effect=fake_read):
                report = run_cycle(config, root / "state.json")
            self.assertEqual(report["new_documents"], 1)
            self.assertEqual(report["rejected_documents"], 1)
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["documents"]), 1)
