import ast
from pathlib import Path
import unittest


class LearningUiTests(unittest.TestCase):
    def test_ui_module_parses(self):
        ast.parse(Path(__file__).with_name("learning_ui.py").read_text(encoding="utf-8"))

    def test_launcher_targets_learning_ui(self):
        launcher = Path(__file__).with_name("run_learning_ui.bat").read_text(encoding="utf-8")
        self.assertIn("learning_ui.py", launcher)


if __name__ == "__main__":
    unittest.main()
