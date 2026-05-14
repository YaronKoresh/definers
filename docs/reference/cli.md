# CLI Reference

Definers exposes a small command surface that is intentionally split between built-in workflow commands and registered launcher surfaces.

## Entry Points

- `definers`
- `python -m definers`

## Core Commands

```bash
definers --version
definers start chat
definers start audio-mastering
definers install --list
definers install audio
definers music-video input.wav 1920 1080 30
definers lyric-video song.wav background.mp4 lyrics.txt bottom
```

## Command Families

- `start <project>` launches a registered surface.
- `install <target>` installs optional runtime dependencies or model assets.
- `music-video` renders a visualizer video from audio.
- `lyric-video` renders a lyric video from audio, background, and lyrics.

## Registered Launcher Surfaces

In addition to the built-in command families, Definers currently exposes registered launcher names directly as CLI commands. Examples include:

- `chat`
- `translate`
- `animation`
- `faiss`
- `audio-mastering`
- `audio-stems`
- `audio-analysis`
- `image-upscale`
- `video-lyrics`
- `train`

These surfaces currently derive from `GUI_LAUNCHERS` in `definers.ui.gui_entrypoints`.

## Stable Defaults

The current defaults are part of the public CLI contract and should not drift silently:

- `start` defaults to `chat`
- `install --type` defaults to `group`
- `lyric-video position` defaults to `bottom`
- `lyric-video --max-dim` defaults to `640`
- `lyric-video --font-size` defaults to `70`
- `lyric-video --text-color` defaults to `white`
- `lyric-video --stroke-color` defaults to `black`
- `lyric-video --stroke-width` defaults to `2`
- `lyric-video --fade` defaults to `0.5`

## Target CLI Architecture

Definers is standardizing the CLI around these rules:

- one authoritative command-definition source should own names, defaults, help text, and handler binding
- built-in commands and registered launcher surfaces should share one declaration model instead of parallel registries
- runtime discovery of launcher surfaces should remain explicit and deterministic
- parser construction, request shaping, and dispatch should stay thin once command definitions are centralized

The authoritative command catalog lives in `definers.cli.command_registry`. Parser construction lives in `definers.cli.parser`, runtime binding lives in `definers.cli.runtime`, and CLI entry assembly lives in `definers.cli.dispatch`.

`definers.cli.command_parser` and `definers.cli.command_dispatcher` stay focused on converting shaped requests into command objects and executing those command objects. The legacy intermediate application package is no longer part of the CLI ownership model.

## Compatibility Rules

- Keep both CLI entry points working.
- Preserve the current command names unless a migration note says otherwise.
- Treat registered launcher names as part of the public command surface while they are exposed directly.
- Preserve install kinds: `group`, `task`, `module`, `model-domain`, `model-task`.
- Fail unknown commands explicitly.
- New internal imports should prefer the direct owner modules under `definers.cli`.