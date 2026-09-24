import unittest

from timdr_answer_engine import answer


class AnswerEngineTests(unittest.TestCase):
    def test_known_answer_is_source_backed(self):
        result = answer("Czy MC K-G jest potwierdzony?")
        self.assertEqual(result["verdict"], "REJECTED_CURRENT_FORM")
        self.assertEqual(result["record_ids"], ["mc-kg-rejected"])

    def test_unknown_answer_refuses(self):
        result = answer("Jak upiec chleb na zakwasie?")
        self.assertFalse(result["is_in_scope"])
        self.assertEqual(result["answer"], "INCONCLUSIVE_NO_MATCH")


if __name__ == "__main__":
    unittest.main()
