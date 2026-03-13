import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="definers")
    parser.add_argument("--version", action="version", version=__version__)

    if argv:
        first = argv[0]
        known = [
            "start",
            "translate",
            "animation",
            "image",
            "chat",
            "faiss",
            "video",
            "audio",
            "train",
            "music-video",
            "lyric-video",
            "--help",
            "--version",
        ]
        if first not in known and not first.startswith("-"):
            print(f"unknown command {first}")
            return 1
    subparsers = parser.add_subparsers(dest="command")

    p_start = subparsers.add_parser("start", help="launch a GUI by name")
    p_start.add_argument(
        "project", nargs="?", default="chat", help="project to launch"
    )

    for name in [
        "translate",
        "animation",
        "image",
        "chat",
        "faiss",
        "video",
        "audio",
        "train",
    ]:
        subparsers.add_parser(name, help=f"launch the {name} interface")

    p_mv = subparsers.add_parser(
        "music-video", help="create a music visualizer video"
    )
    p_mv.add_argument("audio", help="input audio file path")
    p_mv.add_argument("width", type=int, help="video width")
    p_mv.add_argument("height", type=int, help="video height")
    p_mv.add_argument("fps", type=int, help="frames per second")

    p_lv = subparsers.add_parser("lyric-video", help="create a lyric video")
    p_lv.add_argument("audio", help="input audio file")
    p_lv.add_argument("background", help="background video/image")
    p_lv.add_argument("lyrics", help="lyrics text or file")
    p_lv.add_argument(
        "position", choices=["top", "center", "bottom"], default="bottom"
    )
    p_lv.add_argument("--max-dim", type=int, default=640)
    p_lv.add_argument("--font-size", type=int, default=70)
    p_lv.add_argument("--text-color", default="white")
    p_lv.add_argument("--stroke-color", default="black")
    p_lv.add_argument("--stroke-width", type=int, default=2)
    p_lv.add_argument("--fade", type=float, default=0.5)

    from .chat import lyric_video, music_video, start

    args = parser.parse_args(argv)
    cmd = args.command
    if cmd in (None, "start"):
        return start(args.project)
    if cmd in [
        "translate",
        "animation",
        "image",
        "chat",
        "faiss",
        "video",
        "audio",
        "train",
    ]:
        return start(cmd)
    if cmd == "music-video":
        print(music_video(args.audio, args.width, args.height, args.fps))
        return 0
    if cmd == "lyric-video":
        lyrics_text = args.lyrics
        try:
            with open(lyrics_text, encoding="utf-8") as f:
                lyrics_text = f.read()
        except Exception:
            pass
        print(
            lyric_video(
                args.audio,
                args.background,
                lyrics_text,
                args.position,
                max_dim=args.max_dim,
                font_size=args.font_size,
                text_color=args.text_color,
                stroke_color=args.stroke_color,
                stroke_width=args.stroke_width,
                fade_duration=args.fade,
            )
        )
        return 0

    print(f"unknown command {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
