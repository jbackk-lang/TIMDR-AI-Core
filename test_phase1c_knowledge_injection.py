import json
import unittest

from examples.phase1c_knowledge_injection import DATA, trajectory, verify_manifest


class Phase1CTests(unittest.TestCase):
    def setUp(self):
        self.capsule = json.loads(DATA.read_text(encoding="utf-8"))

    def test_manifest_is_frozen_and_matches_sources(self):
        self.assertEqual(verify_manifest(self.capsule), [])

    def test_every_record_declares_scope_and_limit(self):
        for record in self.capsule["records"]:
            for field in ("object", "status", "scope", "limitation", "source"):
                self.assertTrue(record[field].strip(), (record["id"], field))
            self.assertIn(record["source"], self.capsule["source_manifest"])

    def test_holdout_routes_to_required_evidence(self):
        for row in self.capsule["holdout"]:
            result = trajectory(row["question"], self.capsule)
            self.assertEqual(result["status"], "EVIDENCE_READY")
            self.assertTrue(set(row["required_record_ids"]) <= set(result["selected_record_ids"]))

    def test_negative_controls_have_no_evidence(self):
        for question in self.capsule["negative_controls"]:
            self.assertEqual(trajectory(question, self.capsule)["status"], "INCONCLUSIVE_NO_MATCH")

    def test_rejected_bridge_does_not_select_established_bridge(self):
        result = trajectory("Jaki status ma MC K-G w obecnej konfiguracji?", self.capsule)
        self.assertEqual(result["selected_record_ids"], ["mc-kg-rejected"])


if __name__ == "__main__":
    unittest.main()
