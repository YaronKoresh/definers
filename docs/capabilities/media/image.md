# Image Workflows

Definers includes image-oriented workflows for feature extraction, generation support, upscaling, and output preparation.

## Start Here

- `definers.image`
- focused launcher surfaces such as `image-generate`, `image-upscale`, and `image-title`
- guided-job launcher surface: `image-generate-jobs`

## Best Fit

- image feature extraction
- upscale pipelines
- image generation surfaces
- title-card and annotation workflows

## Guided Job Surface

Use `image-generate-jobs` when you want a persistent job folder that can resume across generation, upscaling, and title overlay stages. Keep the narrower `image-generate`, `image-upscale`, and `image-title` launchers for the faster single-step path.

Use this capability when the project centers on still-image transformation or production.