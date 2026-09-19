# PROTECT90_WAVEFORM_MULTICLASS_v0.1 — INVALIDATED

This preregistration draft must not be used for a learning run.

Reason: before it was created, a single training-split episode (`sample_id=4`)
was opened for technical schema inspection (table shape, column names and time
range). The v0.1 draft did not declare that inspection and therefore made an
incorrect claim that no waveform file had been opened.

No waveform training, calibration, holdout access, model selection, or result
reporting was performed under v0.1. The corrective plan is
`PROTECT90_WAVEFORM_MULTICLASS_v0.2.json`; it declares the inspection and uses
the same frozen sample split.
