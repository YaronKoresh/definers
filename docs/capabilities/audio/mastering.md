# Audio Mastering

Definers includes a full mastering workflow for tonal control, dynamics, delivery checks, and reporting.

## Start Here

- Python modules under `definers.audio.mastering`
- Focused launcher surface: `audio-mastering`
- Guided-job launcher surface: `audio-mastering-jobs`

## What It Covers

- spectral balance and reference matching
- loudness and true-peak analysis
- delivery verification
- finalization and mastering reports

## Guided Job Surface

Use `audio-mastering-jobs` when the workflow benefits from a persistent job folder, resumable stages, intermediate stem artifacts, and a saved mastering report. Keep `audio-mastering` as the faster direct surface when you only need the one-pass path.

The guided stem-aware path now exposes dedicated final-pass controls for vocal and other glue reverb, drum edge shaping, and extra vocal pullback, and the saved mastering report includes a Stem Final Pass section with the applied values and loudness-recovery telemetry.

Use this path when the goal is release-oriented mastering rather than raw audio cleanup or stem separation.