import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import skill_guided_learning


class SkillGuidedLearningTests(unittest.TestCase):
    def test_routes_cached_document_without_changing_claim_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            state = {
                "schema": "timdr-online-learning-state/1",
                "documents": {
                    "one": {
                        "title": "Modal vibration analysis",
                        "url": "https://example.org/paper.txt",
                        "sha256": "a" * 64,
                        "top_terms": {"vibration": 4, "modal": 3, "frequency": 2},
                    }
                },
                "terms": {},
            }
            (cache / "online_learning_state.json").write_text(json.dumps(state), encoding="utf-8")
            with patch.object(skill_guided_learning, "run_cycle", return_value={"new_documents": 0, "refreshed_documents": 0, "total_documents": 1, "holdout_accessed": False}):
                result = skill_guided_learning.run_skill_guided_cycle("unused.json", cache)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(result["candidate_cards"], 1)
            self.assertFalse(report["claim_graph_modified"])
            self.assertFalse(report["holdout_accessed"])
            self.assertEqual(report["candidate_cards"][0]["status"], "CANDIDATE_READING_ONLY")
            self.assertEqual(report["candidate_cards"][0]["candidate_branches"][0]["branch"], "K modal")
