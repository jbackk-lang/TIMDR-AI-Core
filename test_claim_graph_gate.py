import unittest

from claim_graph_gate import decide, gate_candidate, render


class ClaimGraphGateTests(unittest.TestCase):
    def test_weingarten_requires_surface(self):
        d = decide("Czy Weingarten działa bezpośrednio na samej krzywej?")
        self.assertEqual(d.verdict, "DEFINED_WITH_LIMIT")
        self.assertIn("g-weingarten", render(d))

    def test_rejected_bridge_cannot_be_confirmed(self):
        d = decide("Czy MC K-G jest potwierdzony?")
        self.assertEqual(d.verdict, "REJECTED_CURRENT_FORM")
        self.assertFalse(gate_candidate(d, "MC K-G jest potwierdzony. Werdykt: REJECTED_CURRENT_FORM Evidence: [mc-kg-rejected]")["accepted"])

    def test_ms_g_bridge_wins_over_broad_mc_kg_pattern(self):
        d = decide("Czy MC M/S-G jest selektorem?")
        self.assertEqual(d.node_id, "mc-msg-diagnostic")
        self.assertEqual(d.verdict, "ESTABLISHED_DIAGNOSTIC_DOMAIN_QUALIFIED")

    def test_deterministic_render_passes_its_own_gate(self):
        d = decide("Czy wspólny czas Chronoprocesu zamienia M/S G i K w jeden operator?")
        self.assertTrue(gate_candidate(d, render(d))["accepted"])

    def test_gia_path_must_stay_frozen_for_evaluation(self):
        d = decide("Czy GIA można ponownie dopasować PCA na holdoucie?")
        self.assertEqual(d.node_id, "gia-frozen-path")
        self.assertEqual(d.verdict, "FROZEN_REFERENCE_REQUIRED")
        self.assertTrue(gate_candidate(d, render(d))["accepted"])

    def test_ai_is_epistemic_filter_not_empirical_authority(self):
        d = decide("Czy AI TIMDR może samo uznać wynik za SUPPORTED?")
        self.assertEqual(d.node_id, "ai-epistemic-filter")
        self.assertEqual(d.verdict, "EPISTEMIC_FILTER_ONLY")
        self.assertTrue(gate_candidate(d, render(d))["accepted"])

    def test_unknown_has_exact_refusal(self):
        d = decide("Jak upiec chleb na zakwasie?")
        self.assertEqual(d.verdict, "INCONCLUSIVE_NO_MATCH")
        self.assertTrue(gate_candidate(d, render(d))["accepted"])
        self.assertFalse(gate_candidate(d, "Może spróbuj przepisu.")["accepted"])


if __name__ == "__main__":
    unittest.main()
