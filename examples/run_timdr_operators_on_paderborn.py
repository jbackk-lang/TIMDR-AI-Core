"""Real-data smoke check for timdr_operators.py -- not a TIMDR result.

Motivation: timdr_operators.py ships with unit tests only against synthetic
sine waves. This script runs the same, unmodified functions
(fft_dominant_mode, winding_number, crossing_number, is_resonant) on real
vibration_1 windows from the frozen Paderborn train/calibration split, to
confirm they produce sane, finite output on the actual signal shape this
repo cares about -- not just on a clean synthetic tone.

This is deliberately NOT a TIMDR test: no preregistration, no controls, no
verdict, no statistic. It only prints numbers for human inspection. It must
never be cited as evidence for or against any TIMDR branch.

Split discipline: reuses paderborn_extractor._members(), which enforces the
frozen-prereg state check and unconditionally refuses "holdout". This script
never touches holdout, directly or indirectly.

Archive reading: paderborn_extractor.py shells out to `tar -xOf`, which
needs bsdtar (libarchive) to understand the .rar container. That binary is
not present in every environment this repo runs in (confirmed absent here).
This script reads the same archives with the pure-Python `unrar-cffi`
package instead, purely so this real-data check can run anywhere -- it does
not replace or modify paderborn_extractor.py, and production extraction
should still go through that module.

Usage:
    pip install --break-system-packages unrar-cffi scipy numpy
    python examples/run_timdr_operators_on_paderborn.py [repo_root]
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paderborn_extractor import SAMPLE_RATE_HZ, WINDOW_SAMPLES, PaderbornExtractionError, _members
from timdr_operators import Modality, crossing_number, fft_dominant_mode, is_resonant, winding_number


def _read_vibration_window(root: Path, archive: str, member: str, window_index: int):
    import numpy as np
    from scipy.io import loadmat
    from unrar.cffi import rarfile

    source = root / "data" / "paderborn_candidate" / "raw" / archive
    rf = rarfile.RarFile(str(source))
    raw = rf.read(member)
    payload = loadmat(BytesIO(raw), squeeze_me=True, struct_as_record=False)
    variable_names = [name for name in payload if not name.startswith("__")]
    if len(variable_names) != 1:
        raise PaderbornExtractionError(f"Unexpected MATLAB top-level schema in {member}.")
    channels = payload[variable_names[0]].Y
    vibration = next((channel.Data for channel in channels if str(channel.Name) == "vibration_1"), None)
    if vibration is None:
        raise PaderbornExtractionError(f"Channel vibration_1 is absent from {member}.")
    values = np.asarray(vibration, dtype=float).reshape(-1)
    start = window_index * WINDOW_SAMPLES
    return values[start:start + WINDOW_SAMPLES]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    train_members = _members(root, "train")
    print(f"Authorized train members: {len(train_members)} (holdout not accessible from here)")

    first, second = train_members[0], train_members[1]
    window_a = _read_vibration_window(root, first["archive"], first["member"], window_index=0)
    window_b = _read_vibration_window(root, second["archive"], second["member"], window_index=0)
    print(f"Window A: {first['archive']}:{first['member']} ({len(window_a)} samples @ {SAMPLE_RATE_HZ} Hz)")
    print(f"Window B: {second['archive']}:{second['member']} ({len(window_b)} samples @ {SAMPLE_RATE_HZ} Hz)")

    f_a, phi_a, a_a = fft_dominant_mode(window_a, fs=SAMPLE_RATE_HZ)
    f_b, phi_b, a_b = fft_dominant_mode(window_b, fs=SAMPLE_RATE_HZ)
    print(f"fft_dominant_mode(A) = f={f_a:.2f} Hz, phi={phi_a:.4f} rad, A={a_a:.6g}")
    print(f"fft_dominant_mode(B) = f={f_b:.2f} Hz, phi={phi_b:.4f} rad, A={a_b:.6g}")

    winding_a, winding_b = winding_number(window_a), winding_number(window_b)
    print(f"winding_number(A)={winding_a:.4f}  winding_number(B)={winding_b:.4f}")

    # crossing_number is O(n^2) memory (see its docstring) -- a full 64000-sample
    # Paderborn window would attempt a ~61 GB allocation and is refused outright.
    # This is a real scaling limit found by running this script, not a synthetic
    # test artifact -- report it honestly instead of silently cropping and moving on.
    crop = 3000
    try:
        crossing_full = crossing_number(window_a)
    except Exception as exc:  # noqa: BLE001 -- want to show the exact refusal message
        print(f"crossing_number(A, full {len(window_a)}-sample window) refused as expected: {exc}")
        crossing_a = crossing_number(window_a[:crop])
        crossing_b = crossing_number(window_b[:crop])
        print(f"crossing_number(A[:{crop}])={crossing_a:.1f}  crossing_number(B[:{crop}])={crossing_b:.1f}  "
              f"(first {crop} samples only, NOT representative of the full window)")
    else:
        print(f"crossing_number(A)={crossing_full:.1f}")

    m_a, m_b = Modality(f=f_a, phi=phi_a, A=a_a), Modality(f=f_b, phi=phi_b, A=a_b)
    print(f"is_resonant(A, B) with default eps (1e-6, 1e-6) = {is_resonant(m_a, m_b)}  "
          "(two different real machines/loads -- False is the expected, sane answer here, "
          "not a claim about the K branch)")

    print("\nSmoke check only: confirms the ported operators run and return finite, "
          "plausible numbers on real vibration data. Not a preregistered test and "
          "not evidence for or against any TIMDR hypothesis.")


if __name__ == "__main__":
    main()
