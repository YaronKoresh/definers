from __future__ import annotations

from definers.ui.apps.surface_hub import SurfaceCard, launch_surface_hub

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
    SurfaceCard(
        launcher="audio-support",
        title="Help",
        description="Open the embedded support assistant when you need a quick explanation of an audio workflow.",
        outcomes=(
            "Ask what each workflow does",
            "Get step-by-step usage help",
            "Stay inside the audio domain",
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

TRAIN_SURFACE_CARDS = (
    SurfaceCard(
        launcher="ml-health",
        title="Health",
        description="Check runtime readiness and inspect the ML capability map before you execute anything.",
        outcomes=(
            "Refresh the live health report",
            "Validate runtime readiness",
            "Review the capability map",
        ),
    ),
    SurfaceCard(
        launcher="ml-train",
        title="Train",
        description="Build one training plan and run one training flow without inference or text tooling on screen.",
        outcomes=(
            "Load local or remote data",
            "Preview the training route",
            "Train and export an artifact",
        ),
    ),
    SurfaceCard(
        launcher="ml-run",
        title="Run",
        description="Run predictions, task inference, or answer generation from one execution-only surface.",
        outcomes=(
            "Predict from saved artifacts",
            "Run task-based inference",
            "Use the answer runtime",
        ),
    ),
    SurfaceCard(
        launcher="ml-text",
        title="Text Lab",
        description="Handle text features, reconstruction, summarization, and prompt optimization without training controls.",
        outcomes=(
            "Extract and reconstruct text features",
            "Run summary flows",
            "Optimize prompts for downstream models",
        ),
    ),
    SurfaceCard(
        launcher="ml-ops",
        title="Ops",
        description="Bootstrap models, inspect checkpoints, resolve languages, and run K-means advice from one support surface.",
        outcomes=(
            "Initialize model files or runtime models",
            "Suggest cluster counts",
            "Resolve checkpoints and language codes",
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


def _launch_train_surface(
    visible_sections,
    *,
    app_title: str,
    heading: str,
    description: str,
):
    from definers.ui.apps.train import launch_train_app

    return launch_train_app(
        visible_sections=visible_sections,
        app_title=app_title,
        hero_label=app_title,
        hero_heading=heading,
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


def launch_audio_hub():
    return launch_surface_hub(
        app_title="Definers Audio",
        eyebrow="Audio Task Hub",
        title="Open one focused audio workflow at a time.",
        description="Choose the job you want to do first. Each surface trims the control set to one task family so non-expert users do not need to scan the entire studio.",
        cards=AUDIO_SURFACE_CARDS,
        legacy_command="definers start audio-workbench",
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


def launch_audio_support_surface():
    return _launch_audio_surface(
        ("Support Chat",),
        app_title="Definers Audio Help",
        description="Ask the built-in helper what each audio workflow does and how to use it.",
    )


def launch_video_hub():
    return launch_surface_hub(
        app_title="Definers Video",
        eyebrow="Video Task Hub",
        title="Choose the exact video workflow you need.",
        description="Video surfaces are now split by outcome so non-expert users do not need to scan composer, lyric, and visualizer controls together.",
        cards=VIDEO_SURFACE_CARDS,
        legacy_command="definers start video-workbench",
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


def launch_image_hub():
    return launch_surface_hub(
        app_title="Definers Image",
        eyebrow="Image Task Hub",
        title="Pick one image job and stay on that track.",
        description="Image surfaces are now split so non-expert users can generate, upscale, or title images without hopping through unrelated controls.",
        cards=IMAGE_SURFACE_CARDS,
        legacy_command="definers start image-workbench",
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


def launch_train_hub():
    return launch_surface_hub(
        app_title="Definers ML",
        eyebrow="ML Task Hub",
        title="Enter the ML workflow you actually need.",
        description="The ML launcher is now split into training, runtime, text, health, and ops surfaces so non-expert users can avoid the full cockpit until they need it.",
        cards=TRAIN_SURFACE_CARDS,
        legacy_command="definers start train-workbench",
    )


def launch_train_workbench():
    from definers.ui.apps.train import launch_train_app

    return launch_train_app(
        app_title="Definers ML Workbench",
        hero_label="Definers ML Workbench",
        hero_heading="Train, run, inspect, and bootstrap models from the full cockpit.",
        hero_description="All ML flows remain available together here for power users who want the original all-in-one workbench.",
    )


def launch_ml_health_surface():
    return _launch_train_surface(
        ("studio",),
        app_title="Definers ML Health",
        heading="Inspect runtime readiness before you train or run models.",
        description="Refresh the live health report, validate runtime readiness, and review the capability map from one narrow surface.",
    )


def launch_ml_train_surface():
    return _launch_train_surface(
        ("train",),
        app_title="Definers ML Train",
        heading="Build one training plan and run one training flow.",
        description="Load local or remote data, preview the route, and export a model artifact without inference or text tooling in view.",
    )


def launch_ml_run_surface():
    return _launch_train_surface(
        ("run",),
        app_title="Definers ML Run",
        heading="Predict, infer, or answer from one execution surface.",
        description="Use saved artifacts, task-based inference, and the answer runtime without opening training or text tooling.",
    )


def launch_ml_text_surface():
    return _launch_train_surface(
        ("text",),
        app_title="Definers ML Text Lab",
        heading="Work on text features, summaries, and prompt shaping only.",
        description="Extract text features, reconstruct text, summarize content, and optimize prompts from one text-only surface.",
    )


def launch_ml_ops_surface():
    return _launch_train_surface(
        ("ops",),
        app_title="Definers ML Ops",
        heading="Bootstrap models and inspect runtime support data.",
        description="Initialize model files, load runtime models, inspect checkpoints, resolve languages, and run K-means advice from one support surface.",
    )


__all__ = [
    "launch_audio_analysis_surface",
    "launch_audio_cleanup_surface",
    "launch_audio_create_surface",
    "launch_audio_hub",
    "launch_audio_mastering_surface",
    "launch_audio_midi_surface",
    "launch_audio_stems_surface",
    "launch_audio_support_surface",
    "launch_audio_vocals_surface",
    "launch_audio_workbench",
    "launch_image_generate_surface",
    "launch_image_hub",
    "launch_image_title_surface",
    "launch_image_upscale_surface",
    "launch_image_workbench",
    "launch_ml_health_surface",
    "launch_ml_ops_surface",
    "launch_ml_run_surface",
    "launch_ml_text_surface",
    "launch_ml_train_surface",
    "launch_train_hub",
    "launch_train_workbench",
    "launch_video_composer_surface",
    "launch_video_hub",
    "launch_video_lyrics_surface",
    "launch_video_visualizer_surface",
    "launch_video_workbench",
]
