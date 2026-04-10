# CLI Reference

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

The authoritative parser lives in `definers.cli.parser`.