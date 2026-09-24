import json
import unittest
from pathlib import Path

from examples.phase1b_knowledge_injection import DATA, trajectory


class Phase1BTests(unittest.TestCase):
    def setUp(self):
        self.capsule = json.loads(DATA.read_text(encoding="utf-8"))

    def test_holdout_records_are_retrieved(self):
        for row in self.capsule["holdout"]:
            result = trajectory(row["question"], self.capsule)
            self.assertEqual(result["status"], "EVIDENCE_READY")
            self.assertTrue(set(row["required_record_ids"]) <= set(result["selected_record_ids"]))

    def test_unknown_question_is_inconclusive(self):
        result = trajectory("Jak upiec chleb na zakwasie?", self.capsule)
        self.assertEqual(result["status"], "INCONCLUSIVE_NO_MATCH")


if __name__ == "__main__":
    unittest.main()
