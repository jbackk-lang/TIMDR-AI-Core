"""Audyt twierdzen README (claim_audit.py): dowod, pokrycie liczb, swiezosc, kompletnosc, kotwica, sformulowania."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import claim_audit as ca  # noqa: E402

README = """# Projekt

## Wynik

Detektor trafil 18 z 20 strzalow (90%) i dziala tak samo jak metoda B. Szybkie zaniki to dysrupcje.
Drugi przebieg dal 7 wykryc.

---

## Inne
"""


def spec_for(tmp: Path, **over):
    (tmp / "README.md").write_text(README, encoding="utf-8")
    (tmp / "wyniki.txt").write_text("trafione=18 z 20\nK1 NIESPELNIONE\n", encoding="utf-8")
    (tmp / "kod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp / "claims.py").write_text("# karty\n", encoding="utf-8")
    (tmp / "prereg.md").write_text("reguly\n", encoding="utf-8")
    hit = lambda: ca.Result(ca.POTWIERDZONE, "18/20") if "trafione=18 z 20" in (tmp / "wyniki.txt").read_text() \
        else ca.Result(ca.SPRZECZNE, "?")
    base = dict(TITLE="test", REPO=str(tmp), README="README.md", PREREG="prereg.md", OUTPUT="a.md",
                SCOPES=[(r"^## Wynik", r"^---")], __file__=str(tmp / "claims.py"),
                CLAIMS=[ca.Claim("T1", "Detektor trafil 18 z 20 strzalow (90%)", hit),
                        ca.Claim("T2", "zdanie, ktorego nie ma", hit)],
                FROZEN={"kod.py": ca.sha256(tmp / "kod.py")},
                COMPLETENESS=[("wyniki.txt", r"NIESPELNIONE", r"niespełn|nie przesz", "K1 niespelnione")],
                ANCHORS=[("prereg.md", "wyniki.txt")],
                FORBIDDEN=[(r"szybki\w*\s+zanik\w*[^.]{0,20}\bto\b[^.]{0,10}dysrupc\w*", r"nie wiadomo|czy", "brak etykiet")],
                ABSOLUTE=[(r"\btak samo\b", "podaj roznice")])
    base.update(over)
    return SimpleNamespace(**base)


class ClaimAuditTests(unittest.TestCase):
    def run_spec(self, **over):
        with TemporaryDirectory() as d:
            spec = spec_for(Path(d), **over)
            text, res = ca.run(spec)
            return text, {c.id: r.verdict for c, r in res["rows"]}, res["findings"]

    def test_claim_verdicts(self):
        _, v, _ = self.run_spec()
        self.assertEqual(v["T1"], ca.POTWIERDZONE)
        self.assertEqual(v["T2"], ca.NIEROZSTRZYGNIETE)  # cytat nie istnieje w README

    def test_uncovered_numbers(self):
        _, _, f = self.run_spec()
        r3 = [x for x in f if x[0] == "R3 pokrycie"]
        self.assertEqual(len(r3), 1)
        self.assertIn("7", r3[0][2])

    def test_completeness_missing_and_given(self):
        _, _, f = self.run_spec()
        self.assertIn(("BRAK KOMPLETNOŚCI"), [x[1] for x in f if x[0] == "R5 kompletność"])
        global README
        old = README
        try:
            README = README.replace("Drugi przebieg", "K1 nie przeszlo. Drugi przebieg")
            _, _, f2 = self.run_spec()
            self.assertEqual([x[1] for x in f2 if x[0] == "R5 kompletność"], [ca.POTWIERDZONE])
        finally:
            README = old

    def test_forbidden_and_absolute_wording(self):
        _, _, f = self.run_spec()
        kinds = [x[0] for x in f]
        self.assertIn("R7a zakazane", kinds)
        self.assertIn("R7b sformułowanie", kinds)

    def test_negated_forbidden_phrase_passes(self):
        global README
        old = README
        try:
            README = README.replace("Szybkie zaniki to dysrupcje.", "Nie wiadomo, czy szybkie zaniki to dysrupcje.")
            _, _, f = self.run_spec()
            self.assertNotIn("R7a zakazane", [x[0] for x in f])
        finally:
            README = old

    def test_stale_hash_and_anchor_outside_git(self):
        with TemporaryDirectory() as d:
            spec = spec_for(Path(d))
            (Path(d) / "kod.py").write_text("x = 2\n", encoding="utf-8")
            _, res = ca.run(spec)
            f = res["findings"]
        self.assertIn(ca.NIEROZSTRZYGNIETE, [x[1] for x in f if x[0] == "R4 świeżość"])
        self.assertIn(ca.NIEROZSTRZYGNIETE, [x[1] for x in f if x[0] == "R6 kotwica"])

    def test_failing_card_is_not_evidence(self):
        boom = ca.Claim("T3", "Detektor trafil 18 z 20 strzalow (90%)", lambda: 1 / 0)
        _, v, _ = self.run_spec(CLAIMS=[boom])
        self.assertEqual(v["T3"], ca.NIEROZSTRZYGNIETE)

    def test_helpers(self):
        self.assertTrue(ca.within(4.5, 4.9, approx=True, pct=True, decimals=1))
        self.assertFalse(ca.within(4.5, 5.6, approx=True, pct=True, decimals=1))
        self.assertTrue(ca.within(50, 56, approx=True))
        self.assertFalse(ca.within(2.4, 2.46, approx=False, decimals=1))
        self.assertAlmostEqual(ca.fisher_exact(3, 1, 1, 3), 0.4857, places=3)
        self.assertEqual([n for _, _, n in ca.numbers("level2 ma 579 i 4,5% oraz K1")], ["579", "4,5"])


if __name__ == "__main__":
    unittest.main()
