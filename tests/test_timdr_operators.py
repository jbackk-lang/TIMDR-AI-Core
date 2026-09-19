"""Plumbing tests: confirm the ported operators work and wire into the
existing LTR layer classes -- not a re-validation of their discriminative
power, which lives in the source repos (see timdr_operators.py docstrings)."""
from __future__ import annotations

import math

import pytest

from timdr_ai_core import FundamentalModelLTR, LayerMModal, LayerTTopology
from timdr_operators import (
    Modality,
    TimdrOperatorsError,
    crossing_number,
    fft_dominant_mode,
    is_resonant,
    winding_number,
)

numpy = pytest.importorskip("numpy")


def test_is_resonant_needs_no_numpy_and_is_strict():
    a = Modality(f=10.0, phi=0.0, A=1.0)
    b = Modality(f=10.0, phi=0.0, A=99.0)  # amplitude is irrelevant to Axiom 5
    assert is_resonant(a, b, eps_f=1e-6, eps_phi=1e-6)
    c = Modality(f=10.0 + 2e-6, phi=0.0, A=1.0)  # clearly past eps_f: not resonant
    assert not is_resonant(a, c, eps_f=1e-6, eps_phi=1e-6)


def test_fft_dominant_mode_recovers_known_sine():
    fs = 1000.0
    n = 1000
    f_true, phi_true, a_true = 25.0, 0.7, 2.0
    t = [i / fs for i in range(n)]
    window = [a_true * math.sin(2 * math.pi * f_true * ti + phi_true) for ti in t]
    f_est, phi_est, a_est = fft_dominant_mode(window, fs)
    assert abs(f_est - f_true) < 1e-6
    assert abs(a_est - a_true) < 1e-6
    assert abs(((phi_est - phi_true + math.pi) % (2 * math.pi)) - math.pi) < 1e-3


def test_fft_dominant_mode_rejects_short_window():
    with pytest.raises(TimdrOperatorsError):
        fft_dominant_mode([1.0, 2.0], fs=100.0)


def test_winding_number_nonzero_on_a_closed_loop_and_zero_on_flat_signal():
    n = 400
    loop = [math.sin(2 * math.pi * i / n) + 0.001 * math.sin(2 * math.pi * 3 * i / n) for i in range(n)]
    assert winding_number(loop) > 0.5
    flat = [0.0] * 50
    assert winding_number(flat) == 0.0  # zero-std guard, not a crash


def test_crossing_number_runs_and_is_nonnegative():
    n = 300
    signal = [math.sin(2 * math.pi * 5 * i / n) + 0.3 * math.sin(2 * math.pi * 17 * i / n) for i in range(n)]
    assert crossing_number(signal) >= 0.0


def test_crossing_number_refuses_oversized_window_with_clear_error():
    """Found by running the ported operators on a real 64000-sample Paderborn
    vibration window (1s @ 64kHz): the vectorized O(n^2) implementation would
    attempt a ~61 GB allocation and crash with a bare MemoryError. It must
    instead fail loudly and clearly, before attempting the allocation. Uses a
    small n + small max_length here so the test itself stays cheap -- the
    real-world trigger (n=64000, default max_length=3000) is documented in
    crossing_number's own docstring and examples/run_timdr_operators_on_paderborn.py."""
    n = 5000
    signal = [math.sin(2 * math.pi * 5 * i / n) for i in range(n)]
    with pytest.raises(TimdrOperatorsError, match="too long"):
        crossing_number(signal, max_length=1000)


def test_layer_m_modal_wired_with_fft_dominant_mode():
    fs = 500.0
    n = 500
    window = [math.sin(2 * math.pi * 40.0 * i / fs) for i in range(n)]
    layer = LayerMModal(transform=lambda w: fft_dominant_mode(w, fs=fs))
    f_est, phi_est, a_est = layer.forward(window)
    assert abs(f_est - 40.0) < 1e-6


def test_layer_t_topology_wired_with_winding_and_crossing():
    n = 400
    loop = [math.sin(2 * math.pi * i / n) for i in range(n)]
    layer = LayerTTopology(transform=lambda w: (winding_number(w), crossing_number(w)))
    winding, crossing = layer.forward(loop)
    assert winding > 0.5
    assert crossing >= 0.0


def test_fundamental_model_ltr_can_now_combine_t_and_m_on_the_same_raw_window():
    """Regression test for a real architectural limitation found while
    wiring these operators in (2026-09-19): the original FundamentalModelLTR
    was a SEQUENTIAL single-argument pipe (T's output became I's input became
    M's input, etc.), so T (topology) and M (modal) -- independent feature
    extractors that both need the *same raw window* -- could not be wired in
    together; M would receive T's (winding, crossing) tuple instead of the
    raw window and fail with TimdrOperatorsError.

    FundamentalModelLTR was redesigned so T and M both read the raw input
    directly and in parallel (see its docstring in timdr_ai_core.py). This
    test proves the fix: both operators now run successfully on the SAME raw
    window in one model.forward() call, each producing its own correct,
    independently-verifiable representation."""
    fs = 500.0
    n = 500
    window = [math.sin(2 * math.pi * 40.0 * i / fs) for i in range(n)]

    model = FundamentalModelLTR(
        topology=LayerTTopology(transform=lambda w: (winding_number(w), crossing_number(w))),
        modal=LayerMModal(transform=lambda w: fft_dominant_mode(w, fs=fs)),
    )
    result = model.forward(window)

    winding, crossing = result["T"]
    f_est, phi_est, a_est = result["M"]
    assert winding > 0.5
    assert crossing >= 0.0
    assert abs(f_est - 40.0) < 1e-6
    # I/It/R default to identity, so E (also default identity) combines the
    # SAME T/M representations checked above, plus R's pass-through of them.
    assert result["E"]["T"] == result["T"]
    assert result["E"]["M"] == result["M"]
