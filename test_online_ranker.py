import json
import tempfile
import unittest
from pathlib import Path

from online_ranker import rank_state
from research_queue import build_queue


class OnlineRankerTests(unittest.TestCase):
    def test_catalogue_route_beats_missing_readme_keywords(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = {"documents": {"modal": {
                "title": "jbackk-lang/TIMDR-Modal-Formalism README",
                "url": "https://example.org/readme",
                "top_terms": {"introduction": 1}, "sha256": "a" * 64,
            }}}
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            ranking = rank_state(state_path)
            item = ranking["documents"][0]
            self.assertEqual(item["declared_branch"], "K modal")
            self.assertEqual(item["repository_role"], "formalization")
            ranking_path = root / "ranking.json"
            ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
            queue = build_queue(ranking_path, state_path)
            self.assertEqual(queue["entries"][0]["branch"], "K modal")
