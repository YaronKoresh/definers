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
- adaptive pre-limiter true-peak trim that relaxes for punch-heavy material
- anti-distortion saturation and soft-clip ceilings across presets
- post-limiter micro-dynamics recovery for repeated drum hits and transient punch
- input-fingerprint-aware finalization that branches for melody-led, sparse, and percussive material
- stem-aware remix balancing that can lift melody carriers and pull drums back when the source profile demands it

## Staged Job Mode

Use `audio-mastering` when the workflow benefits from either a one-pass render or a persistent staged job with resumable steps, intermediate stem artifacts, and a saved mastering report.

When stem-aware mastering is enabled, `audio-mastering` exposes dedicated final-pass controls for vocal and other glue reverb, drum edge and expand-compress shaping, and extra vocal pullback. The vocal glue path now keeps a shorter late tail, the other stem can be pushed denser and more melodic, and drum finishing can run with stronger edge plus extra level lead. Wave 2 adds content-aware routing on top of that: melody-led and sparse inputs reduce clip push and follow-up gain, while percussive inputs keep stronger punch recovery. The same material profile also feeds the stem remix pass so guitar, piano, and other can step forward when melody presence would otherwise get buried. The same surface also exposes staged job actions for prepare, resume, and full-run behavior. The saved mastering report includes a Stem Final Pass section with the applied values and loudness-recovery telemetry.

Lossy delivery verification now retries by attenuating the export instead of re-expanding it back to the ceiling, so decoded true-peak safety takes priority over loudness ambition when a codec overshoots.

Use this path when the goal is release-oriented mastering rather than raw audio cleanup or stem separation.