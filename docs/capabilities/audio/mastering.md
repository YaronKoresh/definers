# Audio Mastering

Definers includes a full mastering workflow for tonal control, dynamics, delivery checks, and reporting.

## Start Here

- Python modules under `definers.audio.mastering`
- Focused launcher surface: `audio-mastering`

## What It Covers

- spectral balance and reference matching
- loudness and true-peak analysis
- delivery verification
- finalization and mastering reports

## Staged Job Mode

Use `audio-mastering` when the workflow benefits from either a one-pass render or a persistent staged job with resumable steps, intermediate stem artifacts, and a saved mastering report.

When stem-aware mastering is enabled, `audio-mastering` exposes dedicated final-pass controls for vocal and other glue reverb, drum edge and expand-compress shaping, and extra vocal pullback. The same surface also exposes staged job actions for prepare, resume, and full-run behavior. The saved mastering report includes a Stem Final Pass section with the applied values and loudness-recovery telemetry.

Use this path when the goal is release-oriented mastering rather than raw audio cleanup or stem separation.