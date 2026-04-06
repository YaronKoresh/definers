from definers.file_ops import catch
from definers.presentation.gui_registry import register_gui_launchers
from definers.presentation.launchers import create_gui_project_starter


def _gui_translate():
    from definers.presentation.apps.translate import launch_translate_app

    return launch_translate_app()


def _gui_animation():
    from definers.presentation.apps.animation import launch_animation_app

    return launch_animation_app()


def _gui_image():
    from definers.presentation.apps.image import launch_image_app

    return launch_image_app()


def _gui_chat():
    from definers.presentation.apps.chat_app import launch_chat_app

    return launch_chat_app()


def _gui_faiss():
    from definers.presentation.apps.faiss import launch_faiss_app

    return launch_faiss_app()


def _gui_video():
    from definers.presentation.apps.video import launch_video_app

    return launch_video_app()


def _gui_audio():
    from definers.presentation.apps.audio import launch_audio_app

    return launch_audio_app()


def _gui_train():
    from definers.presentation.apps.train import launch_train_app

    return launch_train_app()


GUI_LAUNCHERS = register_gui_launchers(
    {
        "translate": "_gui_translate",
        "animation": "_gui_animation",
        "image": "_gui_image",
        "chat": "_gui_chat",
        "faiss": "_gui_faiss",
        "video": "_gui_video",
        "audio": "_gui_audio",
        "train": "_gui_train",
    },
    namespace=globals(),
)


def music_video(audio_path, width=1920, height=1080, fps=30):
    from definers.presentation.music_video_service import (
        music_video as run_music_video,
    )

    return run_music_video(audio_path, width=width, height=height, fps=fps)


def init_stable_whisper():
    from definers.presentation.lyric_video_service import (
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
    from definers.presentation.lyric_video_service import (
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
