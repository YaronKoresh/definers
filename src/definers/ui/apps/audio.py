from definers.ui.gradio_shared import init_chat, launch_blocks


def launch_audio_app(
    tool_names=None,
    *,
    app_title="Definers Audio",
    hero_eyebrow="Production Workspace",
    hero_description="Master, clean, split, generate, and publish audio from one focused workspace.",
    default_tool=None,
    navigation_label="Choose Workflow",
):
    from html import escape

    import gradio as gr

    from definers.audio import (
        audio_to_midi,
        beat_visualizer,
        change_audio_speed,
        create_share_links,
        create_spectrum_visualization,
        dj_mix,
        extend_audio,
        generate_music,
        generate_voice,
        get_audio_feedback,
        identify_instruments,
        midi_to_audio,
        pitch_shift_vocals,
        stem_mixer,
        transcribe_audio,
    )
    from definers.constants import language_codes
    from definers.cuda import device
    from definers.file_ops import save_temp_text as save_text_to_file
    from definers.ml import (
        convert_vocal_rvc,
    )
    from definers.text import random_string
    from definers.ui.apps.audio_app_services import (
        MASTERING_PROFILE_CHOICES,
        STEM_MODEL_STRATEGY_CHOICES,
        describe_stem_model_choice,
        get_mastering_profile_ui_state,
        is_custom_stem_model_strategy,
        run_audio_analysis_tool,
        run_audio_preview_tool,
        run_autotune_song_tool,
        run_compact_audio_tool,
        run_humanize_vocals_tool,
        run_mastering_tool,
        run_remove_silence_tool,
        run_split_audio_tool,
        run_stem_separation_tool,
    )
    from definers.ui.apps.audio_workspace import (
        AUDIO_FORMAT_CHOICES,
        AUDIO_TOOL_MAP,
        get_audio_language_choices,
        prepare_audio_workspace,
        train_voice_lab_model,
    )
    from definers.ui.lyric_video_service import lyric_video
    from definers.ui.music_video_service import music_video

    prepare_audio_workspace()

    available_tool_names = tuple(AUDIO_TOOL_MAP)
    selected_tool_names = tuple(tool_names or available_tool_names)
    tool_map = {
        tool_name: AUDIO_TOOL_MAP[tool_name]
        for tool_name in selected_tool_names
        if tool_name in AUDIO_TOOL_MAP
    }
    if not tool_map:
        tool_map = {
            tool_name: AUDIO_TOOL_MAP[tool_name]
            for tool_name in available_tool_names
        }
    default_tool_name = (
        default_tool
        if isinstance(default_tool, str) and default_tool in tool_map
        else next(iter(tool_map))
    )
    format_choices = AUDIO_FORMAT_CHOICES
    language_choices = get_audio_language_choices(language_codes)
    initial_mastering_state = get_mastering_profile_ui_state(
        MASTERING_PROFILE_CHOICES[0]
    )
    initial_stem_strategy_note = describe_stem_model_choice(
        STEM_MODEL_STRATEGY_CHOICES[0]
    )

    with gr.Blocks(title=app_title) as app:
        gr.HTML(
            f"""<div id=\"header\" class=\"audio-hero\"><p class=\"eyebrow\">{escape(hero_eyebrow)}</p><h1>{escape(app_title)}</h1><p>{escape(hero_description)}</p></div>"""
        )

        with gr.Row(elem_id="nav-dropdown-wrapper"):
            nav_dropdown = gr.Dropdown(
                choices=list(tool_map.keys()),
                value=default_tool_name,
                label=navigation_label,
                elem_id="nav-dropdown",
                visible=len(tool_map) > 1,
            )

        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="main-content"):
                with gr.Group(
                    visible=True, elem_classes="tool-container"
                ) as view_enhancer:
                    gr.Markdown("## Mastering Studio")
                    with gr.Row():
                        with gr.Column():
                            enhancer_input = gr.Audio(
                                label="Upload Mix", type="filepath"
                            )
                            enhancer_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value="WAV",
                            )
                            enhancer_preset = gr.Dropdown(
                                MASTERING_PROFILE_CHOICES,
                                label="Mastering Strategy",
                                value=str(initial_mastering_state["label"]),
                            )
                            enhancer_profile_note = gr.Markdown(
                                value=str(
                                    initial_mastering_state["description"]
                                )
                            )
                            enhancer_stem_mastering = gr.Checkbox(
                                label="Use stem-aware mastering",
                                value=True,
                            )
                            with gr.Accordion("Macro Controls", open=True):
                                enhancer_macro_note = gr.Markdown(
                                    value=str(
                                        initial_mastering_state["macro_note"]
                                    )
                                )
                                enhancer_bass = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["bass"]),
                                    step=0.05,
                                    label="Bass",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                                enhancer_volume = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["volume"]),
                                    step=0.05,
                                    label="Volume",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                                enhancer_effects = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["effects"]),
                                    step=0.05,
                                    label="Effects",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                            with gr.Accordion("Stem-Aware Path", open=False):
                                with gr.Group(
                                    visible=True
                                ) as enhancer_stem_settings:
                                    enhancer_stem_strategy = gr.Dropdown(
                                        STEM_MODEL_STRATEGY_CHOICES,
                                        label="Stem Separation Strategy",
                                        value=STEM_MODEL_STRATEGY_CHOICES[0],
                                    )
                                    enhancer_custom_stem_model = gr.Textbox(
                                        label="Custom separator checkpoint",
                                        placeholder="Example: htdemucs_6s or custom_model.yaml",
                                        visible=False,
                                    )
                                    enhancer_stem_strategy_note = gr.Markdown(
                                        value=initial_stem_strategy_note
                                    )
                                    enhancer_stem_shifts = gr.Slider(
                                        1,
                                        8,
                                        2,
                                        step=1,
                                        label="Stem Separation Shifts",
                                    )
                                    enhancer_stem_mix_headroom = gr.Slider(
                                        3.0,
                                        12.0,
                                        6.0,
                                        step=0.5,
                                        label="Stem Mix Headroom (dB)",
                                    )
                                    enhancer_save_mastered_stems = gr.Checkbox(
                                        label="Save mastered stems",
                                        value=True,
                                    )
                            gr.Markdown(
                                "Auto Analyze chooses a mastering profile after reading the mix. Named profiles lock macro controls to avoid conflict, and Custom Macro Blend unlocks manual shaping."
                            )
                            with gr.Row():
                                enhancer_btn = gr.Button(
                                    "Master Audio", variant="primary"
                                )
                                clear_enhancer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as enhancer_output_box:
                                enhancer_output = gr.Audio(
                                    label="Mastered Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                enhancer_report = gr.File(
                                    label="Mastering Report",
                                    interactive=False,
                                    visible=False,
                                )
                                enhancer_stems_output = gr.File(
                                    label="Mastered Stems",
                                    interactive=False,
                                    file_count="multiple",
                                    visible=False,
                                )
                                enhancer_diagnostics = gr.Markdown()
                                enhancer_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_vocal_finish:
                    gr.Markdown("## Vocal Finishing")
                    with gr.Tabs():
                        with gr.TabItem("AutoTune"):
                            with gr.Row():
                                with gr.Column():
                                    autotune_input = gr.Audio(
                                        label="Upload Song", type="filepath"
                                    )
                                    autotune_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="WAV",
                                    )
                                    autotune_strength = gr.Slider(
                                        0.0,
                                        1.0,
                                        0.7,
                                        step=0.05,
                                        label="Pitch Correction Strength",
                                    )
                                    autotune_correct_timing = gr.Checkbox(
                                        label="Correct timing against detected beat grid",
                                        value=True,
                                    )
                                    autotune_quantize = gr.Slider(
                                        4,
                                        32,
                                        16,
                                        step=4,
                                        label="Beat Grid Density",
                                    )
                                    autotune_tolerance = gr.Slider(
                                        0,
                                        50,
                                        15,
                                        step=1,
                                        label="Tolerance (cents)",
                                    )
                                    autotune_attack = gr.Slider(
                                        0.0,
                                        20.0,
                                        0.1,
                                        step=0.1,
                                        label="Attack Smoothing (ms)",
                                    )
                                    with gr.Row():
                                        autotune_btn = gr.Button(
                                            "AutoTune Song",
                                            variant="primary",
                                        )
                                        clear_autotune_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as autotune_output_box:
                                        autotune_output = gr.Audio(
                                            label="AutoTuned Song",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        autotune_share_links = gr.Markdown()
                        with gr.TabItem("Humanize Vocals"):
                            with gr.Row():
                                with gr.Column():
                                    humanize_input = gr.Audio(
                                        label="Upload Vocal Take",
                                        type="filepath",
                                    )
                                    humanize_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="WAV",
                                    )
                                    humanize_amount = gr.Slider(
                                        0.0,
                                        1.0,
                                        0.5,
                                        step=0.05,
                                        label="Variation Amount",
                                    )
                                    with gr.Row():
                                        humanize_btn = gr.Button(
                                            "Humanize Vocals",
                                            variant="primary",
                                        )
                                        clear_humanize_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as humanize_output_box:
                                        humanize_output = gr.Audio(
                                            label="Humanized Vocals",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        humanize_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_cleanup:
                    gr.Markdown("## Audio Cleanup")
                    with gr.Tabs():
                        with gr.TabItem("Remove Silence"):
                            with gr.Row():
                                with gr.Column():
                                    silence_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    silence_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="WAV",
                                    )
                                    with gr.Row():
                                        silence_btn = gr.Button(
                                            "Remove Silence",
                                            variant="primary",
                                        )
                                        clear_silence_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as silence_output_box:
                                        silence_output = gr.Audio(
                                            label="Silence-Reduced Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        silence_share_links = gr.Markdown()
                        with gr.TabItem("Compact Audio"):
                            with gr.Row():
                                with gr.Column():
                                    compact_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    compact_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    gr.Markdown(
                                        "Creates a lighter export using the project's compact-audio preset."
                                    )
                                    with gr.Row():
                                        compact_btn = gr.Button(
                                            "Compact Audio",
                                            variant="primary",
                                        )
                                        clear_compact_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as compact_output_box:
                                        compact_output = gr.Audio(
                                            label="Compacted Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        compact_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_preview_split:
                    gr.Markdown("## Preview & Split")
                    with gr.Tabs():
                        with gr.TabItem("Smart Preview"):
                            with gr.Row():
                                with gr.Column():
                                    preview_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    preview_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="WAV",
                                    )
                                    preview_duration = gr.Slider(
                                        5,
                                        60,
                                        30,
                                        step=1,
                                        label="Preview Length (seconds)",
                                    )
                                    with gr.Row():
                                        preview_btn = gr.Button(
                                            "Create Preview",
                                            variant="primary",
                                        )
                                        clear_preview_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as preview_output_box:
                                        preview_output = gr.Audio(
                                            label="Preview Clip",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        preview_summary = gr.Markdown()
                                        preview_share_links = gr.Markdown()
                        with gr.TabItem("Split Audio"):
                            with gr.Row():
                                with gr.Column():
                                    split_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    split_format = gr.Radio(
                                        format_choices,
                                        label="Chunk Format",
                                        value="MP3",
                                    )
                                    split_duration = gr.Slider(
                                        5,
                                        300,
                                        30,
                                        step=1,
                                        label="Chunk Length (seconds)",
                                    )
                                    split_skip_time = gr.Number(
                                        label="Skip Time Before First Chunk (seconds)",
                                        value=0,
                                        precision=2,
                                    )
                                    split_chunks_limit = gr.Number(
                                        label="Maximum Number of Chunks (0 = all)",
                                        value=0,
                                        precision=0,
                                    )
                                    split_sample_rate = gr.Number(
                                        label="Target Sample Rate (0 = keep original)",
                                        value=0,
                                        precision=0,
                                    )
                                    with gr.Row():
                                        split_btn = gr.Button(
                                            "Split Audio",
                                            variant="primary",
                                        )
                                        clear_split_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as split_output_box:
                                        split_preview_output = gr.Audio(
                                            label="First Chunk Preview",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        split_files_output = gr.File(
                                            label="Chunk Files",
                                            interactive=False,
                                            file_count="multiple",
                                        )
                                        split_summary_output = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_midi_tools:
                    gr.Markdown("## MIDI Tools")
                    with gr.Tabs():
                        with gr.TabItem("Audio to MIDI"):
                            with gr.Row():
                                with gr.Column():
                                    a2m_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    with gr.Row():
                                        a2m_btn = gr.Button(
                                            "Convert to MIDI", variant="primary"
                                        )
                                        clear_a2m_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as a2m_output_box:
                                        a2m_output = gr.File(
                                            label="Output MIDI",
                                            interactive=False,
                                        )
                                        a2m_share_links = gr.Markdown()
                        with gr.TabItem("MIDI to Audio"):
                            with gr.Row():
                                with gr.Column():
                                    m2a_input = gr.File(
                                        label="Upload MIDI",
                                        file_types=[".mid", ".midi"],
                                    )
                                    m2a_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value=format_choices[0],
                                    )
                                    with gr.Row():
                                        m2a_btn = gr.Button(
                                            "Convert to Audio",
                                            variant="primary",
                                        )
                                        clear_m2a_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as m2a_output_box:
                                        m2a_output = gr.Audio(
                                            label="Output Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        m2a_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_audio_extender:
                    gr.Markdown("## Audio Extender")
                    with gr.Row():
                        with gr.Column():
                            extender_input = gr.Audio(
                                label="Upload Audio to Extend", type="filepath"
                            )
                            extender_duration = gr.Slider(
                                5,
                                60,
                                15,
                                step=1,
                                label="Extend Duration (seconds)",
                            )
                            extender_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                extender_btn = gr.Button(
                                    "Extend Audio", variant="primary"
                                )
                                clear_extender_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as extender_output_box:
                                extender_output = gr.Audio(
                                    label="Extended Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                extender_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem_mixer:
                    gr.Markdown("## Stem Mixer")
                    with gr.Row():
                        with gr.Column():
                            stem_mixer_files = gr.File(
                                label="Upload Stems (Drums, Bass, Vocals, etc.)",
                                file_count="multiple",
                                type="filepath",
                            )
                            stem_mixer_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                stem_mixer_btn = gr.Button(
                                    "Mix Stems", variant="primary"
                                )
                                clear_stem_mixer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as stem_mixer_output_box:
                                stem_mixer_output = gr.Audio(
                                    label="Mixed Track",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                stem_mixer_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_feedback:
                    gr.Markdown("## AI Track Feedback")
                    with gr.Row():
                        with gr.Column():
                            feedback_input = gr.Audio(
                                label="Upload Your Track", type="filepath"
                            )
                            with gr.Row():
                                feedback_btn = gr.Button(
                                    "Get Feedback", variant="primary"
                                )
                                clear_feedback_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            feedback_output = gr.Markdown(label="Feedback")
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_instrument_id:
                    gr.Markdown("## Instrument Identification")
                    with gr.Row():
                        with gr.Column():
                            instrument_id_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                instrument_id_btn = gr.Button(
                                    "Identify Instruments", variant="primary"
                                )
                                clear_instrument_id_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            instrument_id_output = gr.Markdown(
                                label="Detected Instruments"
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_video_gen:
                    gr.Markdown("## AI Music Clip Generation")
                    with gr.Row():
                        with gr.Column():
                            video_gen_audio = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                video_gen_btn = gr.Button(
                                    "Generate Video", variant="primary"
                                )
                                clear_video_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as video_gen_output_box:
                                video_gen_output = gr.Video(
                                    label="Generated Clip",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                video_gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_speed:
                    gr.Markdown("## Speed & Pitch")
                    with gr.Row():
                        with gr.Column():
                            speed_input = gr.Audio(
                                label="Upload Track", type="filepath"
                            )
                            speed_factor = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.01,
                                label="Speed Factor",
                            )
                            preserve_pitch = gr.Checkbox(
                                label="Preserve Pitch (higher quality)",
                                value=True,
                            )
                            speed_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                speed_btn = gr.Button(
                                    "Change Speed", variant="primary"
                                )
                                clear_speed_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as speed_output_box:
                                speed_output = gr.Audio(
                                    label="Modified Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                speed_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem:
                    gr.Markdown("## Stem Separation")
                    with gr.Row():
                        with gr.Column():
                            stem_input = gr.Audio(
                                label="Upload Full Mix", type="filepath"
                            )
                            stem_mode = gr.Radio(
                                [
                                    "Acapella (Vocals Only)",
                                    "Karaoke (Instrumental Only)",
                                    "Vocals + Karaoke",
                                    "Mastering Layers (Vocals / Drums / Bass / Other)",
                                ],
                                label="Separation Mode",
                                value="Acapella (Vocals Only)",
                            )
                            stem_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value="WAV",
                            )
                            stem_mode_note = gr.Markdown(
                                value="**Layer Controls:** Layer strategy settings appear only in Mastering Layers mode. Use Vocals + Karaoke when you only need the vocal and instrumental pair."
                            )
                            with gr.Accordion("Layer Controls", open=False):
                                with gr.Group(
                                    visible=False
                                ) as stem_layer_settings:
                                    stem_model_name = gr.Dropdown(
                                        STEM_MODEL_STRATEGY_CHOICES,
                                        label="Layer Separation Strategy",
                                        value=STEM_MODEL_STRATEGY_CHOICES[0],
                                    )
                                    stem_custom_model_name = gr.Textbox(
                                        label="Custom layer checkpoint",
                                        placeholder="Example: htdemucs_6s or custom_model.yaml",
                                        visible=False,
                                    )
                                    stem_layer_strategy_note = gr.Markdown(
                                        value=initial_stem_strategy_note
                                    )
                                    stem_shifts = gr.Slider(
                                        1,
                                        8,
                                        2,
                                        step=1,
                                        label="Layer Separation Shifts",
                                    )
                            with gr.Row():
                                stem_btn = gr.Button(
                                    "Separate Stems", variant="primary"
                                )
                                clear_stem_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as stem_output_box:
                                stem_output = gr.Audio(
                                    label="Primary Stem Preview",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                stem_files_output = gr.File(
                                    label="Stem Files",
                                    interactive=False,
                                    file_count="multiple",
                                )
                                stem_summary_output = gr.Markdown()
                                stem_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_vps:
                    gr.Markdown("## Vocal Pitch Shifter")
                    with gr.Row():
                        with gr.Column():
                            vps_input = gr.Audio(
                                label="Upload Full Song", type="filepath"
                            )
                            vps_pitch = gr.Slider(
                                -12,
                                12,
                                0,
                                step=1,
                                label="Vocal Pitch Shift (Semitones)",
                            )
                            vps_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vps_btn = gr.Button(
                                    "Shift Vocal Pitch", variant="primary"
                                )
                                clear_vps_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vps_output_box:
                                vps_output = gr.Audio(
                                    label="Pitch Shifted Song",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                vps_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_lab:
                    gr.Markdown("## ðŸ”¬ Voice Lab")
                    with gr.Row(visible=False):
                        experiment = gr.Textbox(value=random_string())
                    with gr.Row():
                        inp = gr.File(label="Input", type="filepath")
                        outp = gr.File(
                            label="Output",
                            type="filepath",
                            file_count="multiple",
                        )
                    with gr.Row(visible=False):
                        lvl = gr.Number(
                            label="(re-)training step",
                            value=1,
                            minimum=1,
                            step=1,
                        )
                    with gr.Row():
                        but1 = gr.Button("Train", variant="primary")
                        but1.click(
                            fn=train_voice_lab_model,
                            inputs=[experiment, inp, lvl],
                            outputs=[outp, lvl],
                        )
                        but2 = gr.Button("Convert", variant="primary")
                        but2.click(
                            fn=convert_vocal_rvc,
                            inputs=[experiment, inp],
                            outputs=[outp],
                        )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_dj:
                    gr.Markdown("## DJ AutoMix")
                    with gr.Row():
                        with gr.Column():
                            dj_files = gr.File(
                                label="Upload Audio Tracks",
                                file_count="multiple",
                                type="filepath",
                                allow_reordering=True,
                            )
                            dj_mix_type = gr.Radio(
                                ["Simple Crossfade", "Beatmatched Crossfade"],
                                label="Mix Type",
                                value="Beatmatched Crossfade",
                            )
                            dj_target_bpm = gr.Number(
                                label="Target BPM (Optional)"
                            )
                            dj_transition = gr.Slider(
                                1,
                                15,
                                5,
                                step=1,
                                label="Transition Duration (seconds)",
                            )
                            dj_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                dj_btn = gr.Button(
                                    "Create DJ Mix", variant="primary"
                                )
                                clear_dj_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as dj_output_box:
                                dj_output = gr.Audio(
                                    label="Final DJ Mix",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                dj_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_music_gen:
                    gr.Markdown("## AI Music Generation")
                    if device() == "cpu":
                        gr.Markdown(
                            "<p style='color:orange;text-align:center;'>Running on a CPU. Music generation will be very slow.</p>"
                        )
                    with gr.Row():
                        with gr.Column():
                            gen_prompt = gr.Textbox(
                                lines=4,
                                label="Music Prompt",
                                placeholder="e.g., '80s synthwave, retro, upbeat'",
                            )
                            gen_duration = gr.Slider(
                                5, 30, 10, step=1, label="Duration (seconds)"
                            )
                            gen_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                gen_btn = gr.Button(
                                    "Generate Music",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as gen_output_box:
                                gen_output = gr.Audio(
                                    label="Generated Music",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_gen:
                    gr.Markdown("## AI Voice Generation")
                    with gr.Row():
                        with gr.Column():
                            vg_ref = gr.Audio(
                                label="Reference Audio (Optional tone guide)",
                                type="filepath",
                            )
                            vg_text = gr.Textbox(
                                lines=4,
                                label="Text to Speak",
                                placeholder="Enter the text you want the generated voice to say...",
                            )
                            vg_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vg_btn = gr.Button(
                                    "Generate Voice",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_vg_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vg_output_box:
                                vg_output = gr.Audio(
                                    label="Generated Voice Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                vg_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_analysis:
                    gr.Markdown("## Analysis & Diagnostics")
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Accordion("Advanced Analysis", open=False):
                                analysis_hop_length = gr.Number(
                                    label="Hop Length",
                                    value=1024,
                                    precision=0,
                                )
                                analysis_duration = gr.Number(
                                    label="Analysis Window (seconds, 0 = full track)",
                                    value=0,
                                    precision=2,
                                )
                                analysis_offset = gr.Number(
                                    label="Start Offset (seconds)",
                                    value=0,
                                    precision=2,
                                )
                            with gr.Row():
                                analysis_btn = gr.Button(
                                    "Analyze Audio", variant="primary"
                                )
                                clear_analysis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column(scale=1):
                            with gr.Group(visible=False) as analysis_output_box:
                                analysis_bpm_key_output = gr.Textbox(
                                    label="Detected Key & BPM",
                                    interactive=False,
                                )
                                analysis_diagnostics_output = gr.Markdown()
                                analysis_json_output = gr.File(
                                    label="Analysis Report",
                                    interactive=False,
                                )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stt:
                    gr.Markdown("## Speech-to-Text")
                    with gr.Row():
                        with gr.Column():
                            stt_input = gr.Audio(
                                label="Upload Speech Audio", type="filepath"
                            )
                            stt_language = gr.Dropdown(
                                language_choices,
                                label="Language",
                                value="english",
                            )
                            with gr.Row():
                                stt_btn = gr.Button(
                                    "Transcribe Audio",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_stt_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            stt_output = gr.Textbox(
                                label="Transcription Result",
                                interactive=False,
                                lines=10,
                            )
                            stt_file_output = gr.File(
                                label="Download Transcript",
                                interactive=False,
                                visible=False,
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_spectrum:
                    gr.Markdown("## Spectrum Analyzer")
                    spec_input = gr.Audio(label="Upload Audio", type="filepath")
                    with gr.Row():
                        spec_btn = gr.Button(
                            "Generate Spectrum", variant="primary"
                        )
                        clear_spec_btn = gr.Button("Clear", variant="secondary")
                    spec_output = gr.Image(
                        label="Spectrum Plot", interactive=False
                    )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_beat_vis:
                    gr.Markdown("## Beat Visualizer")
                    with gr.Row():
                        with gr.Column():
                            vis_image_input = gr.Image(
                                label="Upload Image", type="filepath"
                            )
                            vis_audio_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                        with gr.Column():
                            vis_effect = gr.Radio(
                                [
                                    "None",
                                    "Blur",
                                    "Sharpen",
                                    "Contour",
                                    "Emboss",
                                ],
                                label="Image Effect",
                                value="None",
                            )
                            vis_animation = gr.Radio(
                                ["None", "Zoom In", "Zoom Out"],
                                label="Animation Style",
                                value="None",
                            )
                            vis_intensity = gr.Slider(
                                1.05,
                                1.5,
                                1.15,
                                step=0.01,
                                label="Beat Intensity",
                            )
                            with gr.Row():
                                vis_btn = gr.Button(
                                    "Create Beat Visualizer", variant="primary"
                                )
                                clear_vis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                    with gr.Group(visible=False) as vis_output_box:
                        vis_output = gr.Video(
                            label="Visualizer Output",
                            buttons=["download"],
                        )
                        vis_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_lyric_vid:
                    gr.Markdown("## Lyric Video Creator")
                    with gr.Row():
                        with gr.Column():
                            lyric_audio = gr.Audio(
                                label="Upload Song", type="filepath"
                            )
                            lyric_bg = gr.File(
                                label="Upload Background (Image or Video)",
                                type="filepath",
                            )
                            lyric_position = gr.Radio(
                                ["center", "bottom"],
                                label="Text Position",
                                value="bottom",
                            )
                        with gr.Column():
                            lyric_text = gr.Textbox(
                                label="Lyrics",
                                lines=15,
                                placeholder="Enter lyrics here, one line per phrase...",
                            )
                            load_transcript_btn = gr.Button(
                                "Get Lyrics from Audio (via Speech-to-Text)"
                            )
                            lyric_language = gr.Dropdown(
                                language_choices,
                                label="Lyrics language (for Speech-to-Text)",
                                value="english",
                            )
                    with gr.Row():
                        lyric_btn = gr.Button(
                            "Create Lyric Video", variant="primary"
                        )
                        clear_lyric_btn = gr.Button(
                            "Clear", variant="secondary"
                        )
                    with gr.Group(visible=False) as lyric_output_box:
                        lyric_output = gr.Video(
                            label="Lyric Video Output",
                            buttons=["download"],
                        )
                        lyric_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_chatbot:
                    init_chat("Definers Audio support")

        views = {
            "enhancer": view_enhancer,
            "vocal_finish": view_vocal_finish,
            "cleanup": view_cleanup,
            "preview_split": view_preview_split,
            "midi_tools": view_midi_tools,
            "audio_extender": view_audio_extender,
            "stem_mixer": view_stem_mixer,
            "feedback": view_feedback,
            "instrument_id": view_instrument_id,
            "video_gen": view_video_gen,
            "speed": view_speed,
            "stem": view_stem,
            "vps": view_vps,
            "voice_lab": view_voice_lab,
            "dj": view_dj,
            "music_gen": view_music_gen,
            "voice_gen": view_voice_gen,
            "analysis": view_analysis,
            "stt": view_stt,
            "spectrum": view_spectrum,
            "beat_vis": view_beat_vis,
            "lyric_vid": view_lyric_vid,
            "chatbot": view_chatbot,
        }

        def switch_view(selected_tool_name):
            selected_view_key = tool_map[selected_tool_name]
            return {
                view: gr.update(visible=(key == selected_view_key))
                for key, view in views.items()
            }

        nav_dropdown.change(
            fn=switch_view,
            inputs=nav_dropdown,
            outputs=list(views.values()),
        )

        app.load(
            lambda: switch_view(default_tool_name),
            outputs=list(views.values()),
        )

        def build_share_markup(result_path):
            if not result_path:
                return ""
            return create_share_links(
                "definers",
                "audio",
                result_path,
                "Check out this creation from Definers Audio! ðŸŽ¶",
            )

        def create_ui_handler(
            btn, out_el, out_box, out_share, logic_func, *inputs
        ):
            def ui_handler_generator(*args):
                try:
                    result = logic_func(*args)
                    share_html = build_share_markup(result)
                    return (
                        gr.update(value=btn.value, interactive=True),
                        gr.update(visible=True),
                        gr.update(value=result),
                        gr.update(value=share_html),
                    )
                except Exception:
                    return (
                        gr.update(value=btn.value, interactive=True),
                        gr.update(visible=False),
                        gr.update(value=None),
                        gr.update(value=""),
                    )

            btn.click(
                ui_handler_generator,
                inputs=inputs,
                outputs=[btn, out_box, out_el, out_share],
            )

        def update_mastering_profile_ui(
            profile_name,
            bass,
            volume,
            effects,
        ):
            state = get_mastering_profile_ui_state(
                profile_name,
                bass,
                volume,
                effects,
            )
            return (
                gr.update(
                    value=float(state["bass"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(
                    value=float(state["volume"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(
                    value=float(state["effects"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(value=str(state["description"])),
                gr.update(value=str(state["macro_note"])),
            )

        def update_mastering_stem_ui(
            stem_mastering_enabled,
            model_selection,
            model_override,
        ):
            enabled = bool(stem_mastering_enabled)
            strategy_note = (
                describe_stem_model_choice(model_selection, model_override)
                if enabled
                else "**Stem Strategy:** Stem-aware mastering is off. The track will be processed as a single stereo master."
            )
            return (
                gr.update(visible=enabled),
                gr.update(
                    visible=enabled
                    and is_custom_stem_model_strategy(model_selection)
                ),
                gr.update(value=str(strategy_note)),
            )

        def update_stem_layer_ui(
            separation_mode,
            model_selection,
            model_override,
        ):
            uses_layer_controls = (
                separation_mode
                == "Mastering Layers (Vocals / Drums / Bass / Other)"
            )
            mode_note = (
                "**Layer Controls:** Mastering Layers exports vocals, drums, bass, and other as separate files. Choose the separator strategy that best matches the source."
                if uses_layer_controls
                else "**Layer Controls:** These settings are used only in Mastering Layers mode. Use Vocals + Karaoke when you only need the vocal and instrumental pair."
            )
            return (
                gr.update(value=mode_note),
                gr.update(visible=uses_layer_controls),
                gr.update(
                    visible=uses_layer_controls
                    and is_custom_stem_model_strategy(model_selection)
                ),
                gr.update(
                    value=describe_stem_model_choice(
                        model_selection,
                        model_override,
                    )
                ),
            )

        enhancer_preset.change(
            update_mastering_profile_ui,
            [
                enhancer_preset,
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
            ],
            [
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
                enhancer_profile_note,
                enhancer_macro_note,
            ],
        )

        enhancer_stem_mastering.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        enhancer_stem_strategy.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        enhancer_custom_stem_model.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        stem_mode.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        stem_model_name.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        stem_custom_model_name.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        def mastering_ui(
            audio_path,
            output_format,
            profile_name,
            bass,
            volume,
            effects,
            stem_mastering,
            stem_model_name,
            stem_shifts_value,
            stem_mix_headroom_value,
            save_mastered_stems_value,
            stem_model_override,
        ):
            yield {
                enhancer_btn: gr.update(
                    value="Mastering...", interactive=False
                ),
                enhancer_output_box: gr.update(visible=False),
                enhancer_output: None,
                enhancer_report: gr.update(value=None, visible=False),
                enhancer_stems_output: gr.update(value=None, visible=False),
                enhancer_diagnostics: "",
                enhancer_share_links: "",
            }
            try:
                (
                    mastered_path,
                    report_path,
                    diagnostics_text,
                    stem_files,
                ) = run_mastering_tool(
                    audio_path,
                    output_format,
                    profile_name,
                    bass,
                    volume,
                    effects,
                    stem_mastering,
                    stem_model_name,
                    stem_shifts_value,
                    stem_mix_headroom_value,
                    save_mastered_stems_value,
                    stem_model_override,
                )
                yield {
                    enhancer_btn: gr.update(
                        value="Master Audio", interactive=True
                    ),
                    enhancer_output_box: gr.update(visible=True),
                    enhancer_output: mastered_path,
                    enhancer_report: gr.update(
                        value=report_path,
                        visible=report_path is not None,
                    ),
                    enhancer_stems_output: gr.update(
                        value=stem_files or None,
                        visible=bool(stem_files),
                    ),
                    enhancer_diagnostics: diagnostics_text,
                    enhancer_share_links: build_share_markup(mastered_path),
                }
            except Exception as error:
                yield {
                    enhancer_btn: gr.update(
                        value="Master Audio", interactive=True
                    ),
                    enhancer_output_box: gr.update(visible=False),
                    enhancer_output: None,
                    enhancer_report: gr.update(value=None, visible=False),
                    enhancer_stems_output: gr.update(value=None, visible=False),
                    enhancer_diagnostics: "",
                    enhancer_share_links: "",
                }
                raise gr.Error(str(error))

        enhancer_btn.click(
            mastering_ui,
            [
                enhancer_input,
                enhancer_format,
                enhancer_preset,
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_stem_shifts,
                enhancer_stem_mix_headroom,
                enhancer_save_mastered_stems,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_btn,
                enhancer_output_box,
                enhancer_output,
                enhancer_report,
                enhancer_stems_output,
                enhancer_diagnostics,
                enhancer_share_links,
            ],
        )

        create_ui_handler(
            a2m_btn,
            a2m_output,
            a2m_output_box,
            a2m_share_links,
            audio_to_midi,
            a2m_input,
        )
        create_ui_handler(
            m2a_btn,
            m2a_output,
            m2a_output_box,
            m2a_share_links,
            midi_to_audio,
            m2a_input,
            m2a_format,
        )
        create_ui_handler(
            extender_btn,
            extender_output,
            extender_output_box,
            extender_share_links,
            extend_audio,
            extender_input,
            extender_duration,
            extender_format,
        )
        create_ui_handler(
            autotune_btn,
            autotune_output,
            autotune_output_box,
            autotune_share_links,
            run_autotune_song_tool,
            autotune_input,
            autotune_format,
            autotune_strength,
            autotune_correct_timing,
            autotune_quantize,
            autotune_tolerance,
            autotune_attack,
        )
        create_ui_handler(
            humanize_btn,
            humanize_output,
            humanize_output_box,
            humanize_share_links,
            run_humanize_vocals_tool,
            humanize_input,
            humanize_amount,
            humanize_format,
        )
        create_ui_handler(
            silence_btn,
            silence_output,
            silence_output_box,
            silence_share_links,
            run_remove_silence_tool,
            silence_input,
            silence_format,
        )
        create_ui_handler(
            compact_btn,
            compact_output,
            compact_output_box,
            compact_share_links,
            run_compact_audio_tool,
            compact_input,
            compact_format,
        )
        create_ui_handler(
            stem_mixer_btn,
            stem_mixer_output,
            stem_mixer_output_box,
            stem_mixer_share_links,
            stem_mixer,
            stem_mixer_files,
            stem_mixer_format,
        )

        def preview_ui(audio_path, max_duration, output_format):
            yield {
                preview_btn: gr.update(
                    value="Building Preview...", interactive=False
                ),
                preview_output_box: gr.update(visible=False),
                preview_output: None,
                preview_summary: "",
                preview_share_links: "",
            }
            try:
                preview_path, preview_text = run_audio_preview_tool(
                    audio_path,
                    max_duration,
                    output_format,
                )
                yield {
                    preview_btn: gr.update(
                        value="Create Preview", interactive=True
                    ),
                    preview_output_box: gr.update(visible=True),
                    preview_output: preview_path,
                    preview_summary: preview_text,
                    preview_share_links: build_share_markup(preview_path),
                }
            except Exception as error:
                yield {
                    preview_btn: gr.update(
                        value="Create Preview", interactive=True
                    ),
                    preview_output_box: gr.update(visible=False),
                    preview_output: None,
                    preview_summary: "",
                    preview_share_links: "",
                }
                raise gr.Error(str(error))

        preview_btn.click(
            preview_ui,
            [preview_input, preview_duration, preview_format],
            [
                preview_btn,
                preview_output_box,
                preview_output,
                preview_summary,
                preview_share_links,
            ],
        )

        def split_ui(
            audio_path,
            chunk_duration,
            output_format,
            chunks_limit,
            skip_time,
            target_sample_rate,
        ):
            yield {
                split_btn: gr.update(value="Splitting...", interactive=False),
                split_output_box: gr.update(visible=False),
                split_preview_output: None,
                split_files_output: None,
                split_summary_output: "",
            }
            try:
                preview_path, split_files, summary_text = run_split_audio_tool(
                    audio_path,
                    chunk_duration,
                    output_format,
                    chunks_limit,
                    skip_time,
                    target_sample_rate,
                )
                yield {
                    split_btn: gr.update(value="Split Audio", interactive=True),
                    split_output_box: gr.update(visible=True),
                    split_preview_output: preview_path,
                    split_files_output: split_files,
                    split_summary_output: summary_text,
                }
            except Exception as error:
                yield {
                    split_btn: gr.update(value="Split Audio", interactive=True),
                    split_output_box: gr.update(visible=False),
                    split_preview_output: None,
                    split_files_output: None,
                    split_summary_output: "",
                }
                raise gr.Error(str(error))

        split_btn.click(
            split_ui,
            [
                split_input,
                split_duration,
                split_format,
                split_chunks_limit,
                split_skip_time,
                split_sample_rate,
            ],
            [
                split_btn,
                split_output_box,
                split_preview_output,
                split_files_output,
                split_summary_output,
            ],
        )

        create_ui_handler(
            video_gen_btn,
            video_gen_output,
            video_gen_output_box,
            video_gen_share_links,
            music_video,
            video_gen_audio,
        )
        create_ui_handler(
            speed_btn,
            speed_output,
            speed_output_box,
            speed_share_links,
            change_audio_speed,
            speed_input,
            speed_factor,
            preserve_pitch,
            speed_format,
        )

        def stem_ui(
            audio_path,
            separation_mode,
            output_format,
            model_name,
            shifts_value,
            model_override,
        ):
            mode_map = {
                "Acapella (Vocals Only)": "acapella",
                "Karaoke (Instrumental Only)": "karaoke",
                "Vocals + Karaoke": "vocals_karaoke",
                "Mastering Layers (Vocals / Drums / Bass / Other)": "mastering_layers",
            }
            resolved_mode = mode_map.get(separation_mode, "acapella")
            yield {
                stem_btn: gr.update(value="Separating...", interactive=False),
                stem_output_box: gr.update(visible=False),
                stem_output: None,
                stem_files_output: None,
                stem_summary_output: "",
                stem_share_links: "",
            }
            try:
                primary_output, stem_files, summary_text = (
                    run_stem_separation_tool(
                        audio_path,
                        resolved_mode,
                        output_format,
                        model_name,
                        shifts_value,
                        model_override,
                    )
                )
                yield {
                    stem_btn: gr.update(
                        value="Separate Stems", interactive=True
                    ),
                    stem_output_box: gr.update(visible=True),
                    stem_output: primary_output,
                    stem_files_output: stem_files,
                    stem_summary_output: summary_text,
                    stem_share_links: build_share_markup(primary_output),
                }
            except Exception as error:
                yield {
                    stem_btn: gr.update(
                        value="Separate Stems", interactive=True
                    ),
                    stem_output_box: gr.update(visible=False),
                    stem_output: None,
                    stem_files_output: None,
                    stem_summary_output: "",
                    stem_share_links: "",
                }
                raise gr.Error(str(error))

        stem_btn.click(
            stem_ui,
            [
                stem_input,
                stem_mode,
                stem_format,
                stem_model_name,
                stem_shifts,
                stem_custom_model_name,
            ],
            [
                stem_btn,
                stem_output_box,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
            ],
        )

        create_ui_handler(
            vps_btn,
            vps_output,
            vps_output_box,
            vps_share_links,
            pitch_shift_vocals,
            vps_input,
            vps_pitch,
            vps_format,
        )
        create_ui_handler(
            dj_btn,
            dj_output,
            dj_output_box,
            dj_share_links,
            dj_mix,
            dj_files,
            dj_mix_type,
            dj_target_bpm,
            dj_transition,
            dj_format,
        )
        create_ui_handler(
            gen_btn,
            gen_output,
            gen_output_box,
            gen_share_links,
            generate_music,
            gen_prompt,
            gen_duration,
            gen_format,
        )
        create_ui_handler(
            vg_btn,
            vg_output,
            vg_output_box,
            vg_share_links,
            generate_voice,
            vg_text,
            vg_ref,
            vg_format,
        )
        create_ui_handler(
            vis_btn,
            vis_output,
            vis_output_box,
            vis_share_links,
            beat_visualizer,
            vis_image_input,
            vis_audio_input,
            vis_effect,
            vis_animation,
            vis_intensity,
        )
        create_ui_handler(
            lyric_btn,
            lyric_output,
            lyric_output_box,
            lyric_share_links,
            lyric_video,
            lyric_audio,
            lyric_bg,
            lyric_text,
            lyric_position,
        )

        def analysis_ui(audio_path, hop_length, duration_value, offset_value):
            yield {
                analysis_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                analysis_output_box: gr.update(visible=False),
                analysis_bpm_key_output: "",
                analysis_diagnostics_output: "",
                analysis_json_output: None,
            }
            try:
                bpm_key_text, diagnostics_text, report_path = (
                    run_audio_analysis_tool(
                        audio_path,
                        hop_length,
                        duration_value,
                        offset_value,
                    )
                )
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    ),
                    analysis_output_box: gr.update(visible=True),
                    analysis_bpm_key_output: bpm_key_text,
                    analysis_diagnostics_output: diagnostics_text,
                    analysis_json_output: report_path,
                }
            except Exception as error:
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    ),
                    analysis_output_box: gr.update(visible=False),
                    analysis_bpm_key_output: "",
                    analysis_diagnostics_output: "",
                    analysis_json_output: None,
                }
                raise gr.Error(str(error))

        analysis_btn.click(
            analysis_ui,
            [
                analysis_input,
                analysis_hop_length,
                analysis_duration,
                analysis_offset,
            ],
            [
                analysis_btn,
                analysis_output_box,
                analysis_bpm_key_output,
                analysis_diagnostics_output,
                analysis_json_output,
            ],
        )

        def feedback_ui(audio_path):
            yield {
                feedback_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                feedback_output: "",
            }
            try:
                feedback_text = get_audio_feedback(audio_path)
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    ),
                    feedback_output: feedback_text,
                }
            except Exception as error:
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    )
                }
                raise gr.Error(str(error))

        feedback_btn.click(
            feedback_ui, [feedback_input], [feedback_btn, feedback_output]
        )

        def instrument_id_ui(audio_path):
            yield {
                instrument_id_btn: gr.update(
                    value="Identifying...", interactive=False
                ),
                instrument_id_output: "",
            }
            try:
                instrument_text = identify_instruments(audio_path)
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments",
                        interactive=True,
                    ),
                    instrument_id_output: instrument_text,
                }
            except Exception as error:
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments",
                        interactive=True,
                    )
                }
                raise gr.Error(str(error))

        instrument_id_btn.click(
            instrument_id_ui,
            [instrument_id_input],
            [instrument_id_btn, instrument_id_output],
        )

        def stt_ui(audio_path, language):
            yield {
                stt_btn: gr.update(value="Transcribing...", interactive=False),
                stt_output: "",
                stt_file_output: gr.update(visible=False),
            }
            try:
                transcript = transcribe_audio(audio_path, language)
                file_path = save_text_to_file(transcript)
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    ),
                    stt_output: transcript,
                    stt_file_output: gr.update(visible=True, value=file_path),
                }
            except Exception as error:
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    )
                }
                raise gr.Error(str(error))

        stt_btn.click(
            stt_ui,
            [stt_input, stt_language],
            [stt_btn, stt_output, stt_file_output],
        )

        def spec_ui(audio_path):
            yield {
                spec_btn: gr.update(value="Generating...", interactive=False),
                spec_output: None,
            }
            try:
                spec_image = create_spectrum_visualization(audio_path)
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    ),
                    spec_output: spec_image,
                }
            except Exception as error:
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    )
                }
                raise gr.Error(str(error))

        spec_btn.click(spec_ui, [spec_input], [spec_btn, spec_output])

        def clear_ui(*components):
            updates = {}
            for comp in components:
                if isinstance(
                    comp,
                    (
                        gr.Audio,
                        gr.Video,
                        gr.Image,
                        gr.File,
                        gr.Textbox,
                        gr.Markdown,
                    ),
                ):
                    updates[comp] = None
                if isinstance(comp, gr.Group):
                    updates[comp] = gr.update(visible=False)
            return updates

        clear_enhancer_btn.click(
            lambda: clear_ui(
                enhancer_input,
                enhancer_output,
                enhancer_report,
                enhancer_stems_output,
                enhancer_diagnostics,
                enhancer_share_links,
                enhancer_output_box,
            ),
            [],
            [
                enhancer_input,
                enhancer_output,
                enhancer_report,
                enhancer_stems_output,
                enhancer_diagnostics,
                enhancer_share_links,
                enhancer_output_box,
            ],
        )
        clear_autotune_btn.click(
            lambda: clear_ui(
                autotune_input,
                autotune_output,
                autotune_share_links,
                autotune_output_box,
            ),
            [],
            [
                autotune_input,
                autotune_output,
                autotune_share_links,
                autotune_output_box,
            ],
        )
        clear_humanize_btn.click(
            lambda: clear_ui(
                humanize_input,
                humanize_output,
                humanize_share_links,
                humanize_output_box,
            ),
            [],
            [
                humanize_input,
                humanize_output,
                humanize_share_links,
                humanize_output_box,
            ],
        )
        clear_silence_btn.click(
            lambda: clear_ui(
                silence_input,
                silence_output,
                silence_share_links,
                silence_output_box,
            ),
            [],
            [
                silence_input,
                silence_output,
                silence_share_links,
                silence_output_box,
            ],
        )
        clear_compact_btn.click(
            lambda: clear_ui(
                compact_input,
                compact_output,
                compact_share_links,
                compact_output_box,
            ),
            [],
            [
                compact_input,
                compact_output,
                compact_share_links,
                compact_output_box,
            ],
        )
        clear_preview_btn.click(
            lambda: clear_ui(
                preview_input,
                preview_output,
                preview_summary,
                preview_share_links,
                preview_output_box,
            ),
            [],
            [
                preview_input,
                preview_output,
                preview_summary,
                preview_share_links,
                preview_output_box,
            ],
        )
        clear_split_btn.click(
            lambda: clear_ui(
                split_input,
                split_preview_output,
                split_files_output,
                split_summary_output,
                split_output_box,
            ),
            [],
            [
                split_input,
                split_preview_output,
                split_files_output,
                split_summary_output,
                split_output_box,
            ],
        )
        clear_a2m_btn.click(
            lambda: clear_ui(a2m_input, a2m_output, a2m_output_box),
            [],
            [a2m_input, a2m_output, a2m_output_box],
        )
        clear_m2a_btn.click(
            lambda: clear_ui(m2a_input, m2a_output, m2a_output_box),
            [],
            [m2a_input, m2a_output, m2a_output_box],
        )
        clear_extender_btn.click(
            lambda: clear_ui(
                extender_input, extender_output, extender_output_box
            ),
            [],
            [extender_input, extender_output, extender_output_box],
        )
        clear_stem_mixer_btn.click(
            lambda: clear_ui(
                stem_mixer_files,
                stem_mixer_output,
                stem_mixer_output_box,
            ),
            [],
            [stem_mixer_files, stem_mixer_output, stem_mixer_output_box],
        )
        clear_feedback_btn.click(
            lambda: clear_ui(feedback_input, feedback_output),
            [],
            [feedback_input, feedback_output],
        )
        clear_instrument_id_btn.click(
            lambda: clear_ui(instrument_id_input, instrument_id_output),
            [],
            [instrument_id_input, instrument_id_output],
        )
        clear_video_gen_btn.click(
            lambda: clear_ui(
                video_gen_audio, video_gen_output, video_gen_output_box
            ),
            [],
            [video_gen_audio, video_gen_output, video_gen_output_box],
        )
        clear_speed_btn.click(
            lambda: clear_ui(speed_input, speed_output, speed_output_box),
            [],
            [speed_input, speed_output, speed_output_box],
        )
        clear_stem_btn.click(
            lambda: clear_ui(
                stem_input,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
                stem_output_box,
            ),
            [],
            [
                stem_input,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
                stem_output_box,
            ],
        )
        clear_vps_btn.click(
            lambda: clear_ui(vps_input, vps_output, vps_output_box),
            [],
            [vps_input, vps_output, vps_output_box],
        )
        clear_dj_btn.click(
            lambda: clear_ui(dj_files, dj_output, dj_output_box),
            [],
            [dj_files, dj_output, dj_output_box],
        )
        clear_gen_btn.click(
            lambda: {
                **clear_ui(gen_output, gen_output_box),
                **{gen_prompt: ""},
            },
            [],
            [gen_output, gen_output_box, gen_prompt],
        )
        clear_vg_btn.click(
            lambda: {
                **clear_ui(vg_ref, vg_output, vg_output_box),
                **{vg_text: ""},
            },
            [],
            [vg_ref, vg_output, vg_output_box, vg_text],
        )
        clear_analysis_btn.click(
            lambda: {
                **clear_ui(
                    analysis_input,
                    analysis_diagnostics_output,
                    analysis_json_output,
                    analysis_output_box,
                ),
                **{analysis_bpm_key_output: ""},
            },
            [],
            [
                analysis_input,
                analysis_bpm_key_output,
                analysis_diagnostics_output,
                analysis_json_output,
                analysis_output_box,
            ],
        )
        clear_stt_btn.click(
            lambda: clear_ui(stt_input, stt_output, stt_file_output),
            [],
            [stt_input, stt_output, stt_file_output],
        )
        clear_spec_btn.click(
            lambda: clear_ui(spec_input, spec_output),
            [],
            [spec_input, spec_output],
        )
        clear_vis_btn.click(
            lambda: clear_ui(
                vis_image_input, vis_audio_input, vis_output, vis_output_box
            ),
            [],
            [vis_image_input, vis_audio_input, vis_output, vis_output_box],
        )
        clear_lyric_btn.click(
            lambda: {
                **clear_ui(
                    lyric_audio, lyric_bg, lyric_output, lyric_output_box
                ),
                **{lyric_text: ""},
            },
            [],
            [lyric_audio, lyric_bg, lyric_output, lyric_output_box, lyric_text],
        )
        load_transcript_btn.click(
            transcribe_audio,
            [lyric_audio, lyric_language],
            [lyric_text],
        )

    launch_blocks(app)
