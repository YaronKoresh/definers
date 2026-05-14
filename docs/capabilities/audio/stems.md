# Audio Stems

Definers supports stem separation workflows for vocals, accompaniment, and layer-oriented remix preparation.

## Start Here

- Python modules under `definers.audio.stems`
- Focused launcher surface: `audio-stems`

## Best Fit

- vocal extraction
- instrumental isolation
- pre-mix stem workflows
- cleanup before arrangement or mastering

## Cleanup Behavior

Stem cleanup is role-aware. Vocal and melodic material now keeps more low-level detail when the source is sparse or has a delicate decay, while stronger noise-gate and residual suppression still pull down constant low-level bleed. This keeps quiet phrase tails and subtle melodic content from being stripped out during AI cleanup.

Stem separation now also records whether dereverb, denoise, vocal restoration, or instrumental cleanup already ran on the source. When those repaired stems continue into mastering, downstream cleanup backs off instead of re-scrubbing the same content and shaving off phrase tails a second time.

When the separated stems are sent into stem-aware mastering, drums can receive extra edge and level, the other stem can take more glue and density, vocal ambience stays shorter to avoid an echo-like tail, and the remix stage can rebalance roles from the input fingerprint so melody carriers stay present on melody-led material.

Use this capability when the target output is separated material, not a finished master.