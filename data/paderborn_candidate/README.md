# Paderborn candidate dataset

This directory contains only the manifest. The downloaded `.rar` archives are
kept in `raw/` and ignored by Git because they total about 516 MB.

Selected classes are healthy `K001`, artificial outer-ring EDM damage `KA01`,
and artificial inner-ring EDM damage `KI01`. The official Paderborn
documentation identifies `KA01` as outer-ring EDM and `KI01` as inner-ring EDM.

`PADERBORN_CANDIDATE_MANIFEST.json` is an immutable snapshot from the
download stage, not a statement of the current experiment status: its SHA-256
is referenced by the frozen split. Do not edit the manifest to update status.
Later work created `prereg/PADERBORN_MS_REPLICATION_v0.1.json` and
`prereg/PADERBORN_MS_HYPOTHESIS_v0.1.json`. A separate activation probe
reached train/calibration in v0.7 but was INCONCLUSIVE (zero calibration
errors); its holdout remained closed. See
`prereg/ACTIVATION_PROBE_v0.7_CALIBRATION_RESULT.md`.
