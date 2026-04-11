# Training and Retrieval

Definers supports model training, regression utilities, feature handling, and retrieval-oriented runtime pieces such as FAISS-backed flows.

## Start Here

- `definers.ml.training`
- `definers.ml.regression_api`
- `definers.ml.health_api`
- `definers start train`

## Best Fit

- training loops and data feeding
- regression baselines and helpers
- runtime ML health checks
- feature pipelines that lead into retrieval or search workflows

Use this path when the problem is model lifecycle, not only text generation.

## GUI Workflow

Definers exposes one ML training GUI with two modes inside the same `train` workbench.

### Guided Mode

- available inside `definers start train`
- for non-expert users who want one clear entry path without routing dozens of parameters manually
- guided intake starts from three intents: local files or local collection paths, remote dataset, or continue yesterday's model
- the mode now inspects tabular files, media collections, folder-derived labels, and text or tabular sidecars before it unlocks plan preview and training
- ambiguous beginner routes are reduced to one quick decision inside guided mode when Definers can safely narrow the choice instead of forcing an immediate switch to advanced mode
- hosted runtimes such as Hugging Face Spaces and ZeroGPU keep preview inspection, first-pass dataset sizing, media collection sizing, and retention inside smaller hosted-safe budgets
- guided training ends with a `Use Result` step that saves a session manifest, artifact sidecar, and rollout metrics for later recovery

### Advanced Mode

- available inside `definers start train`
- for expert users who want direct control over labels, splits, routing, resume, and validation details

## Guided Mode Contract

The guided mode inside `train` owns these beginner-first rules:

- accept one beginner-safe route at a time, and ask for one quick decision when local files and remote datasets or other narrow ambiguities both appear
- allow a previous `.joblib` model artifact only when new data is also present
- inspect tabular schema before training and infer likely label columns when possible
- inspect local image, audio, and video collections and normalize safe labels from folder names or aligned sidecars when possible
- keep validation mandatory before plan preview and training
- fall back explicitly to the advanced workbench only when the route stays unsafe after the quick decision or when the inputs exceed the safe guided contract

## Guided Recommendations

- guided inspection emits recommendations with both a rationale and a confidence level
- high-confidence defaults are applied automatically for label columns, batch size, validation split, test split, and stratify when the route is safe
- drop-column and row-sampling suggestions stay visible as optional guidance when the system cannot justify applying them automatically
- hosted runtimes can turn large datasets or media collections into an explicit first-pass recommendation and a single quick decision instead of silently attempting the full workload

## Resume Coaching

- when a previous `.joblib` artifact is present, guided mode looks for a saved artifact sidecar and session manifest
- resume review classifies the route as `Continue Safely`, `Re-Fit With Checks`, or `Start Fresh`
- incompatible source families or conflicting label contracts downgrade resume into a fresh start instead of forcing the artifact back into the training path
- compatible previous sessions can restore label hints from yesterday's manifest so the user does not need to remember the old routing choices

## Session Persistence

- each guided train run now writes a managed train session manifest under the train output root
- the final model artifact also receives a nearby manifest sidecar so future guided resume runs can recover the previous request, inspection, recommendations, and next actions
- guided inspection and guided training also record rollout metrics so hosted and beginner-first routing outcomes can be audited without scraping UI output
- the `Use Result` panel points users back to the `Run` tab for prediction, back to guided mode for continued training, or to the saved manifest for recovery and audit

## Runtime Boundaries

- guided validation depends on ML health being ready through `definers.ml.health_api`
- local file inspection trusts the same safe-root policy as the data loader runtime
- guided mode can now keep a dominant local file family for a first pass, align media sidecars, or infer media labels from folder structure when the route is still safe enough for beginner operation
- unsupported or unresolved mixed-modality routes still fall back to the advanced workbench
- hosted runtimes use shorter managed-output retention and can clean guided session outputs aggressively after training completes
- train session manifests and artifact sidecars live under the managed output-path policy rather than the repository working tree
- the advanced workbench remains the escape hatch for manual routing, explicit drop/select controls, and mixed-modality workflows