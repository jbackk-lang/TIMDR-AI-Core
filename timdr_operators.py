"""Ready-made, faithfully-ported operators to plug into the T/I/M/It/R/E layers.

``timdr_ai_core.py`` deliberately keeps ``LayerTTopology``/``LayerMModal``/
``LayerRResonance``/etc. as *generic* adapters -- identity by default, "a
transport default, not a TIMDR operator" (see that module's docstrings).
This module supplies the actual, previously-validated math those layers can
be wired up with via their ``transform=`` constructor argument, so
TIMDR-AI-Core stops being pure scaffolding without inventing any new
mathematics here.

Every function below is a 1:1 port from a sibling repository -- same
formulas, same edge-case handling, only repackaged as a small, dependency-
declared, independently testable unit. None of them establishes a TIMDR
verdict; they only compute representations/features, exactly like every
other layer in ``FundamentalModelLTR``. Provenance is named per function so
a change in the source repo can be tracked back here.

``is_resonant``/``Modality`` need only the standard library. ``fft_dominant_mode``,
``winding_number``, and ``crossing_number`` need numpy (declared under the
``operators`` extra in ``pyproject.toml``) -- importing this module without
numpy installed still works, but calling those three raises a clear
``TimdrOperatorsError`` instead of a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised via TimdrOperatorsError path
    _np = None


class TimdrOperatorsError(RuntimeError):
    pass


def _require_numpy(feature: str):
    if _np is None:
        raise TimdrOperatorsError(
            f"{feature} needs numpy. Install it with: "
            'pip install -e ".[operators]" (or add numpy>=1.24 yourself).'
        )
    return _np


# ---------------------------------------------------------------------------
# R layer -- modal resonance (Axiom 5, K branch).
# Ported unchanged from TIMDR-Modal-Formalism/timdr_modal/phase_sync.py
# (Modality dataclass + is_resonant()). Needs no numpy at all.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Modality:
    """(f, phi, A) -- Axiom 3 of Axioms_K_TIMDR.md, literally."""

    f: float
    phi: float
    A: float = 1.0


def is_resonant(m_i: Modality, m_j: Modality, eps_f: float = 1e-6, eps_phi: float = 1e-6) -> bool:
    """Axiom 5, literally: |f_i-f_j| < eps_f AND |phi_i-phi_j| < eps_phi.

    Strict inequalities, matching the axiom text -- a value sitting exactly
    on the threshold does not count as resonant. See
    ``TIMDR-Modal-Formalism/docs/RESULT_K_GRID_FREQ_v0.1.md`` for a real,
    documented failure mode of this exact strictness (a percentile-based
    epsilon calibration can collapse to 0.0 on a quantized signal, which
    then invalidates even a self-comparison) -- callers computing eps_f/eps_phi
    from data should be aware of that edge case.

    NOTE ON WIRING: this function compares *two* modalities, so it does not
    fit the single-argument ``Transform = Callable[[Any], Any]`` shape that
    ``LayerRResonance`` expects in the current sequential T->I->M->It->R->E
    chain. Do not force it into ``LayerRResonance(transform=...)`` by
    partial-applying one fixed reference modality -- that would silently
    change what R measures based on argument order, which is worse than
    leaving R as identity. Use it directly as a standalone comparison
    utility instead (e.g. between two ``TIMDR_AI_System.run()`` outputs).
    """
    return abs(m_i.f - m_j.f) < eps_f and abs(m_i.phi - m_j.phi) < eps_phi


# ---------------------------------------------------------------------------
# M layer -- dominant-mode (f, phi, A) extraction from a real-valued window.
# Ported unchanged from
# TIMDR-Modal-Formalism/timdr_modal/real_data_validation.py
# (_wrap_to_pi, _fft_peak_f_phi_A). Needs numpy.
# ---------------------------------------------------------------------------


def _wrap_to_pi(x: float) -> float:
    return float((x + pi) % (2.0 * pi) - pi)


def fft_dominant_mode(window: Sequence[float], fs: float) -> tuple[float, float, float]:
    """Dominant-frequency (f, phi, A) from one real-valued window.

    ``phi`` is already in the sine convention of Axiom 4 (see the source
    module's docstring for the cosine->sine derivation) and is the phase at
    sample 0 of *this* window -- converting it to a shared t=0 reference
    across multiple windows is the caller's job (see
    ``_local_phase_to_global`` in the source module if you need that).

    Drop-in transform for ``LayerMModal``:
    ``LayerMModal(transform=lambda w: fft_dominant_mode(w, fs=fs))``.
    """
    np = _require_numpy("fft_dominant_mode")
    n = len(window)
    if n < 4:
        raise TimdrOperatorsError("window too short for FFT (n<4)")
    x = np.asarray(window, dtype=float) - np.mean(window)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if len(spec) < 2:
        raise TimdrOperatorsError("window too short -- no FFT bins beyond DC")
    idx = 1 + int(np.argmax(np.abs(spec[1:])))
    f_peak = float(freqs[idx])
    phi_cos_convention = float(np.angle(spec[idx]))
    phi_local = _wrap_to_pi(phi_cos_convention + pi / 2.0)
    A_peak = float(2.0 * np.abs(spec[idx]) / n)
    return f_peak, phi_local, A_peak


# ---------------------------------------------------------------------------
# T layer -- winding/crossing number via delay embedding.
# Ported unchanged from GIA-TIMDR/core/winding_crossing_ms_bridge.py
# (_embed_3d, project_pca_2d, winding_metric_fn, crossing_metric_fn) and the
# LAG constant from GIA-TIMDR/core/trefoil_ms_bridge.py. Needs numpy.
#
# Honest scope, carried over from the source repo: on synthetic data these
# metrics were REJECTED (1/50 grid cells) as a general-purpose bridge; on
# real data they showed a strong, partially-characterized signal on CWRU
# bearing faults (123/150) but an inconsistent one on seismic/BTC data
# (see GIA-TIMDR/docs/geometry/RESULT_REAL_*_NOISE_ROBUSTNESS.md). Porting
# them here makes them available as a *feature*, not a validated,
# general-purpose classifier signal -- treat any downstream result the same
# way the source repo does (diagnostic, not selector; domain-dependent).
# ---------------------------------------------------------------------------

LAG = 1  # frozen in the source repo -- NOT tuned after seeing a result.


def _embed_3d(signal_1d, lag: int = LAG):
    np = _require_numpy("_embed_3d")
    x = np.asarray(signal_1d, dtype=float)
    std = float(x.std())
    if std < 1e-12:
        return None
    xn = (x - float(x.mean())) / std
    n = len(xn) - 2 * lag
    if n < 4:
        return None
    return np.stack([xn[:n], xn[lag:lag + n], xn[2 * lag:2 * lag + n]], axis=1)


def _project_pca_2d(pts3d):
    np = _require_numpy("_project_pca_2d")
    centered = pts3d - pts3d.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    top2 = eigvecs[:, -2:]
    return centered @ top2


def winding_number(window: Sequence[float]) -> float:
    """Turning number of a delay-embedded, PCA-projected window.

    Drop-in transform for ``LayerTTopology``:
    ``LayerTTopology(transform=winding_number)``.
    """
    np = _require_numpy("winding_number")
    pts3d = _embed_3d(window)
    if pts3d is None:
        return 0.0
    pts2d = _project_pca_2d(pts3d)
    if pts2d is None or len(pts2d) < 3:
        return 0.0
    centered = pts2d - pts2d.mean(axis=0)
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    theta_unwrapped = np.unwrap(theta)
    return float(abs(theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi))


def _cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def crossing_number(window: Sequence[float]) -> float:
    """Count of non-adjacent self-intersections of the same embedded curve.

    Vectorized (all segment pairs at once); mathematically identical to the
    naive double loop in the source repo, changed only for speed on larger
    windows -- see that repo's commit history for the timing that motivated it.
    """
    np = _require_numpy("crossing_number")
    pts3d = _embed_3d(window)
    if pts3d is None:
        return 0.0
    pts2d = _project_pca_2d(pts3d)
    if pts2d is None or len(pts2d) < 4:
        return 0.0

    P, Q = pts2d[:-1], pts2d[1:]
    AB = Q - P
    m = len(P)
    Pi, Qi, Pj, Qj = P[:, None, :], Q[:, None, :], P[None, :, :], Q[None, :, :]
    ABi, ABj = AB[:, None, :], AB[None, :, :]

    o1 = _cross2(ABi, Pj - Pi)
    o2 = _cross2(ABi, Qj - Pi)
    o3 = _cross2(ABj, Pi - Pj)
    o4 = _cross2(ABj, Qi - Pj)

    intersect = (
        (np.sign(o1) != np.sign(o2))
        & (np.sign(o3) != np.sign(o4))
        & (o1 != 0) & (o2 != 0) & (o3 != 0) & (o4 != 0)
    )
    idx_i, idx_j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    mask = idx_j >= idx_i + 2  # non-adjacent segments, no double counting
    return float(np.sum(intersect & mask))
