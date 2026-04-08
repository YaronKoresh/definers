from functools import partial

from definers.file_ops import catch
from definers.ui.gui_registry import register_gui_launchers
from definers.ui.launchers import create_gui_project_starter


def _gui_translate():
    from definers.ui.apps.translate import launch_translate_app

    return launch_translate_app()


def _gui_animation():
    from definers.ui.apps.animation import launch_animation_app

    return launch_animation_app()


def _launch_focused_surface(launcher_name):
    from definers.ui.apps import focused_surfaces

    launcher = getattr(focused_surfaces, launcher_name, None)
    if not callable(launcher):
        raise LookupError(f"No focused GUI launcher called {launcher_name}")
    return launcher()


def _gui_chat():
    from definers.ui.apps.chat_app import launch_chat_app

    return launch_chat_app()


def _gui_faiss():
    from definers.ui.apps.faiss import launch_faiss_app

    return launch_faiss_app()


GUI_LAUNCHERS = register_gui_launchers(
    {
        "translate": _gui_translate,
        "animation": _gui_animation,
        "image": partial(_launch_focused_surface, "launch_image_hub"),
        "image-workbench": partial(
            _launch_focused_surface,
            "launch_image_workbench",
        ),
        "image-generate": partial(
            _launch_focused_surface,
            "launch_image_generate_surface",
        ),
        "image-upscale": partial(
            _launch_focused_surface,
            "launch_image_upscale_surface",
        ),
        "image-title": partial(
            _launch_focused_surface,
            "launch_image_title_surface",
        ),
        "chat": _gui_chat,
        "faiss": _gui_faiss,
        "video": partial(_launch_focused_surface, "launch_video_hub"),
        "video-workbench": partial(
            _launch_focused_surface,
            "launch_video_workbench",
        ),
        "video-composer": partial(
            _launch_focused_surface,
            "launch_video_composer_surface",
        ),
        "video-lyrics": partial(
            _launch_focused_surface,
            "launch_video_lyrics_surface",
        ),
        "video-visualizer": partial(
            _launch_focused_surface,
            "launch_video_visualizer_surface",
        ),
        "audio": partial(_launch_focused_surface, "launch_audio_hub"),
        "audio-workbench": partial(
            _launch_focused_surface,
            "launch_audio_workbench",
        ),
        "audio-mastering": partial(
            _launch_focused_surface,
            "launch_audio_mastering_surface",
        ),
        "audio-vocals": partial(
            _launch_focused_surface,
            "launch_audio_vocals_surface",
        ),
        "audio-cleanup": partial(
            _launch_focused_surface,
            "launch_audio_cleanup_surface",
        ),
        "audio-stems": partial(
            _launch_focused_surface,
            "launch_audio_stems_surface",
        ),
        "audio-analysis": partial(
            _launch_focused_surface,
            "launch_audio_analysis_surface",
        ),
        "audio-create": partial(
            _launch_focused_surface,
            "launch_audio_create_surface",
        ),
        "audio-midi": partial(
            _launch_focused_surface,
            "launch_audio_midi_surface",
        ),
        "audio-support": partial(
            _launch_focused_surface,
            "launch_audio_support_surface",
        ),
        "train": partial(_launch_focused_surface, "launch_train_hub"),
        "train-workbench": partial(
            _launch_focused_surface,
            "launch_train_workbench",
        ),
        "ml-health": partial(
            _launch_focused_surface,
            "launch_ml_health_surface",
        ),
        "ml-train": partial(
            _launch_focused_surface,
            "launch_ml_train_surface",
        ),
        "ml-run": partial(_launch_focused_surface, "launch_ml_run_surface"),
        "ml-text": partial(
            _launch_focused_surface,
            "launch_ml_text_surface",
        ),
        "ml-ops": partial(_launch_focused_surface, "launch_ml_ops_surface"),
    },
    namespace=globals(),
)


def music_video(audio_path, width=1920, height=1080, fps=30):
    from definers.ui.music_video_service import (
        music_video as run_music_video,
    )

    return run_music_video(audio_path, width=width, height=height, fps=fps)


def init_stable_whisper():
    from definers.ui.lyric_video_service import (
        init_stable_whisper as initialize_stable_whisper,
    )

    return initialize_stable_whisper()


def lyric_video(
    audio_path,
    background_path,
    lyrics_text,
    text_position,
    max_dim=640,
    font_size=70,
    text_color="white",
    stroke_color="black",
    stroke_width=2,
    fade_duration=0.5,
):
    from definers.ui.lyric_video_service import (
        lyric_video as run_lyric_video,
    )

    return run_lyric_video(
        audio_path,
        background_path,
        lyrics_text,
        text_position,
        max_dim=max_dim,
        font_size=font_size,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fade_duration=fade_duration,
    )


def start(project: str):
    def on_missing(project_name: str):
        catch(f"Error: No project called '{project_name}' !")

    return create_gui_project_starter(
        globals(),
        on_missing,
        registry=GUI_LAUNCHERS,
    ).start(project)


__all__ = [
    "GUI_LAUNCHERS",
    "init_stable_whisper",
    "lyric_video",
    "music_video",
    "start",
]
