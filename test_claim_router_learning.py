import json
import unittest

from claim_router_learning import DATA, guarded_route, train


class ClaimRouterLearningTests(unittest.TestCase):
    def setUp(self):
        self.dataset = json.loads(DATA.read_text(encoding="utf-8"))
        self.model = train(self.dataset["train"])

    def test_guard_keeps_graph_as_authority(self):
        result = guarded_route(self.model, "Czy Weingarten wymaga powierzchni?")
        self.assertEqual(result["logical_label"], "weingarten-surface")

    def test_unknown_is_not_promoted_by_learning(self):
        result = guarded_route(self.model, "Jak upiec chleb?")
        self.assertEqual(result["logical_label"], "INCONCLUSIVE_NO_MATCH")
        self.assertEqual(result["verdict"], "INCONCLUSIVE_NO_MATCH")

    def test_holdout_is_not_used_by_training(self):
        self.assertNotIn("holdout", train.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
