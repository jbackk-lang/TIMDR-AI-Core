import json,unittest
from claim_router_learning_v03 import PLAN,V1,V2,features,guarded_route,train
class V03(unittest.TestCase):
 def setUp(self):
  a=json.loads(V1.read_text(encoding='utf8'));b=json.loads(V2.read_text(encoding='utf8'));self.model=train(a['train']+a['calibration']+b['extra_train']+b['calibration'])
 def test_holdout_normalization(self):self.assertEqual(features('po holdoucie'),features('po holdout'))
 def test_unknown_graph_authority(self):self.assertEqual(guarded_route(self.model,'Jak ustawić temperaturę piekarnika?')['verdict'],'INCONCLUSIVE_NO_MATCH')
if __name__=='__main__':unittest.main()
