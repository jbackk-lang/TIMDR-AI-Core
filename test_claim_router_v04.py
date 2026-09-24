import unittest
from claim_graph_gate import decide
class V04(unittest.TestCase):
 def test_branch_bound_alias(self):self.assertEqual(decide('Czy wspólny czas M/S i G scala gałęzie?').node_id,'chrono-separate')
 def test_plain_time_is_not_alias(self):self.assertEqual(decide('Czy wspólny czas obiadu jest Chronoprocesem?').verdict,'INCONCLUSIVE_NO_MATCH')
if __name__=='__main__':unittest.main()
