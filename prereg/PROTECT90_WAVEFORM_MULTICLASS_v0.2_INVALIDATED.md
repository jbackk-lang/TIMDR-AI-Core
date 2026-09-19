# PROTECT90_WAVEFORM_MULTICLASS_v0.2 — INVALIDATED

This preregistration must not be used for a learning run.

The first authorized train/calibration attempt stopped at training episode 4:
`Bus_2_Line_02_03B` has a zero RMS baseline. Version v0.2 required division by
that baseline and therefore could not create its fixed feature vector.

No calibration result and no holdout waveform access occurred under v0.2. The
corrective v0.3 hypothesis declares an inactive-channel rule: if baseline RMS
is at most `1e-12`, the affected normalized feature is exactly `0.0` and no
division is performed. This is a new preregistration, not an edit to v0.2.
