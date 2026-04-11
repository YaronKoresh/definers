from __future__ import annotations

from definers.ui.apps.surface_hub import SurfaceCard

AUDIO_SURFACE_CARDS = (
    SurfaceCard(
        launcher="audio-mastering",
        title="Mastering",
        description="Master one mix with guided presets, stem-aware processing, and an export report.",
        outcomes=(
            "Upload one stereo mix",
            "Pick a mastering profile",
            "Export mastered audio and optional stems",
        ),
    ),
    SurfaceCard(
        launcher="audio-vocals",
        title="Vocals",
        description="Tune, humanize, shift, convert, or generate vocals without unrelated production controls.",
        outcomes=(
            "AutoTune a full song",
            "Humanize or pitch-shift vocals",
            "Clone or convert voices",
        ),
    ),
    SurfaceCard(
        launcher="audio-cleanup",
        title="Cleanup",
        description="Trim silence, compact files, split long recordings, build previews, and adjust speed.",
        outcomes=(
            "Remove silence and shrink files",
            "Create previews or chunk exports",
            "Extend or retime audio",
        ),
    ),
    SurfaceCard(
        launcher="audio-stems",
        title="Stems And Mix Prep",
        description="Separate stems, build mastering layers, mix stems together, or prepare DJ transitions.",
        outcomes=(
            "Export vocals or karaoke",
            "Create four mastering layers",
            "Mix stems or assemble DJ blends",
        ),
    ),
    SurfaceCard(
        launcher="audio-analysis",
        title="Analysis",
        description="Inspect BPM and key, request mix feedback, identify instruments, transcribe speech, and view spectrum plots.",
        outcomes=(
            "Run diagnostics and export reports",
            "Identify instruments and transcribe speech",
            "Inspect the spectrum visually",
        ),
    ),
    SurfaceCard(
        launcher="audio-create",
        title="Creation",
        description="Generate music, build beat-reactive visuals, create lyric videos, and synthesize voices from one creative surface.",
        outcomes=(
            "Generate music from text",
            "Create lyric or beat-reactive videos",
            "Synthesize voice lines",
        ),
    ),
    SurfaceCard(
        launcher="audio-midi",
        title="MIDI",
        description="Convert audio to MIDI or render MIDI back to audio without the rest of the studio.",
        outcomes=(
            "Extract MIDI from audio",
            "Render MIDI to audio",
            "Keep the workflow short and direct",
        ),
    ),
)

VIDEO_SURFACE_CARDS = (
    SurfaceCard(
        launcher="video-composer",
        title="Composer",
        description="Build a styled music video with media, layout, overlays, and composition controls only.",
        outcomes=(
            "Upload media and pick a style",
            "Tune overlays and effects",
            "Render one composed video",
        ),
    ),
    SurfaceCard(
        launcher="video-lyrics",
        title="Lyric Video",
        description="Create a lyric video from an audio track and background without the composition workspace around it.",
        outcomes=(
            "Upload audio and background",
            "Paste lyrics and choose placement",
            "Render one lyric video",
        ),
    ),
    SurfaceCard(
        launcher="video-visualizer",
        title="Visualizer",
        description="Generate a music visualizer from one audio file with only export dimensions and frame rate exposed.",
        outcomes=(
            "Upload one track",
            "Set size and frame rate",
            "Export a visualizer video",
        ),
    ),
)

IMAGE_SURFACE_CARDS = (
    SurfaceCard(
        launcher="image-generate",
        title="Generate",
        description="Create one image from text without upscaling or titling controls in the way.",
        outcomes=(
            "Enter a prompt",
            "Choose width and height",
            "Generate one image",
        ),
    ),
    SurfaceCard(
        launcher="image-upscale",
        title="Upscale",
        description="Upload one image and upscale it without having to think about generation controls.",
        outcomes=(
            "Upload an existing image",
            "Run the upscale pass",
            "Download the improved result",
        ),
    ),
    SurfaceCard(
        launcher="image-title",
        title="Add Titles",
        description="Overlay top, middle, and bottom titles on one image with no other image tooling shown.",
        outcomes=(
            "Upload one image",
            "Enter the text lines",
            "Export a titled image",
        ),
    ),
)


def _launch_audio_surface(tool_names, *, app_title: str, description: str):
    from definers.ui.apps.audio import launch_audio_app

    return launch_audio_app(
        tool_names=tool_names,
        app_title=app_title,
        hero_eyebrow="Audio Workflow",
        hero_description=description,
        default_tool=tool_names[0],
        navigation_label="Choose Task",
    )


def _launch_video_surface(visible_tabs, *, app_title: str, description: str):
    from definers.ui.apps.video import launch_video_app

    return launch_video_app(
        visible_tabs=visible_tabs,
        app_title=app_title,
        hero_eyebrow="Video Workflow",
        hero_description=description,
    )


def _launch_image_surface(steps, *, app_title: str, description: str):
    from definers.ui.apps.image import launch_image_app

    return launch_image_app(
        steps=steps,
        app_title=app_title,
        hero_eyebrow="Image Workflow",
        hero_description=description,
    )


def launch_audio_workbench():
    from definers.ui.apps.audio import launch_audio_app

    return launch_audio_app(
        app_title="Definers Audio Workbench",
        hero_eyebrow="Advanced Audio Workbench",
        hero_description="All audio workflows remain available here for power users who want the full production surface.",
        navigation_label="Choose Tool",
    )


def launch_audio_mastering_surface():
    return _launch_audio_surface(
        ("Mastering Studio",),
        app_title="Definers Mastering",
        description="Master one mix with guided presets, stem-aware processing, and a clear export path.",
    )


def launch_audio_vocals_surface():
    return _launch_audio_surface(
        (
            "Vocal Finishing",
            "Vocal Pitch Shifter",
            "Voice Lab",
            "Voice Gen",
        ),
        app_title="Definers Vocal Studio",
        description="Tune, humanize, shift, clone, and convert voices without the rest of the audio production stack on screen.",
    )


def launch_audio_cleanup_surface():
    return _launch_audio_surface(
        (
            "Audio Cleanup",
            "Preview & Split",
            "Audio Extender",
            "Speed & Pitch",
        ),
        app_title="Definers Audio Cleanup",
        description="Trim, compact, preview, split, extend, and retime audio from one repair-oriented surface.",
    )


def launch_audio_stems_surface():
    return _launch_audio_surface(
        ("Stem Separation", "Stem Mixer", "DJ AutoMix"),
        app_title="Definers Stem Studio",
        description="Separate stems, mix them back together, and assemble DJ transitions without unrelated controls.",
    )


def launch_audio_analysis_surface():
    return _launch_audio_surface(
        (
            "Analysis",
            "Track Feedback",
            "Instrument ID",
            "Speech-to-Text",
            "Spectrum",
        ),
        app_title="Definers Audio Analysis",
        description="Inspect one track at a time through diagnostics, feedback, transcription, and spectrum views.",
    )


def launch_audio_create_surface():
    return _launch_audio_surface(
        (
            "Music Gen",
            "Music Clip Generation",
            "Beat Visualizer",
            "Lyric Video",
        ),
        app_title="Definers Audio Creation",
        description="Generate music, create visual outputs, and package audio into simple publishable assets.",
    )


def launch_audio_midi_surface():
    return _launch_audio_surface(
        ("MIDI Tools",),
        app_title="Definers MIDI Tools",
        description="Convert between audio and MIDI without the rest of the production workspace.",
    )


def launch_video_workbench():
    from definers.ui.apps.video import launch_video_app

    return launch_video_app(
        app_title="Definers Video Workbench",
        hero_eyebrow="Advanced Video Workbench",
        hero_description="All video workflows remain available here for users who need the full composer, lyric, and visualizer stack together.",
    )


def launch_video_composer_surface():
    return _launch_video_surface(
        ("composer",),
        app_title="Definers Video Composer",
        description="Build one styled music video with composition and overlay controls only.",
    )


def launch_video_lyrics_surface():
    return _launch_video_surface(
        ("lyrics",),
        app_title="Definers Lyric Video",
        description="Create one lyric video from an audio track and a background without broader video controls.",
    )


def launch_video_visualizer_surface():
    return _launch_video_surface(
        ("visualizer",),
        app_title="Definers Music Visualizer",
        description="Generate a visualizer from one track with only the export dimensions and frame rate shown.",
    )


def launch_image_workbench():
    from definers.ui.apps.image import launch_image_app

    return launch_image_app(
        app_title="Definers Image Workbench",
        hero_eyebrow="Advanced Image Workbench",
        hero_description="Generation, upscaling, and titling remain available together here for users who want the full flow in one place.",
    )


def launch_image_generate_surface():
    return _launch_image_surface(
        ("generate",),
        app_title="Definers Image Generator",
        description="Create one image from a prompt with only prompt and resolution controls exposed.",
    )


def launch_image_upscale_surface():
    return _launch_image_surface(
        ("upscale",),
        app_title="Definers Image Upscale",
        description="Upload one image and upscale it without generation or title controls on the page.",
    )


def launch_image_title_surface():
    return _launch_image_surface(
        ("title",),
        app_title="Definers Image Titles",
        description="Upload one image and add top, middle, or bottom titles without the rest of the image toolchain.",
    )


def launch_train_workbench():
    from definers.ui.apps.train import launch_train_app

    return launch_train_app(
        app_title="Definers ML Workbench",
        hero_label="Definers ML Workbench",
        hero_heading="Train, run, inspect, and bootstrap models from the full cockpit.",
        hero_description="All ML flows remain available together here for users who want an all-in-one workbench.",
    )


__all__ = [
    "launch_audio_analysis_surface",
    "launch_audio_cleanup_surface",
    "launch_audio_create_surface",
    "launch_audio_mastering_surface",
    "launch_audio_midi_surface",
    "launch_audio_stems_surface",
    "launch_audio_vocals_surface",
    "launch_audio_workbench",
    "launch_image_generate_surface",
    "launch_image_title_surface",
    "launch_image_upscale_surface",
    "launch_image_workbench",
    "launch_train_workbench",
    "launch_video_composer_surface",
    "launch_video_lyrics_surface",
    "launch_video_visualizer_surface",
    "launch_video_workbench",
]
