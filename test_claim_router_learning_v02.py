import json
import unittest

from claim_router_learning_v02 import PLAN, V1, guarded_route, tokens, train


class RouterV02Tests(unittest.TestCase):
    def setUp(self):
        v1=json.loads(V1.read_text(encoding="utf-8")); plan=json.loads(PLAN.read_text(encoding="utf-8")); self.model=train(v1["train"]+v1["calibration"]+plan["extra_train"])

    def test_holdout_inflection_normalizes(self): self.assertIn("holdout",tokens("po holdoucie"))
    def test_logic_remains_authority(self): self.assertEqual(guarded_route(self.model,"Jak ugotować ryż?")["verdict"],"INCONCLUSIVE_NO_MATCH")
    def test_holdout_not_loaded(self): self.assertNotIn("holdout",train.__code__.co_varnames)


if __name__=="__main__": unittest.main()
