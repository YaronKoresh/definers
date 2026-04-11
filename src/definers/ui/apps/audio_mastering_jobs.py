from __future__ import annotations


def _mastering_compute_note(stem_mastering_enabled: bool) -> str:
    if stem_mastering_enabled:
        return (
            "**Compute Profile:** Prepare Job is light, Separate Stems is the heavy separator stage, "
            "Build Stem Mix reuses the saved stems, and Finalize Master renders the delivery file and report."
        )
    return (
        "**Compute Profile:** Stereo-only guided mastering stays on the lighter path. "
        "Prepare Job analyzes the mix, then Finalize Master renders the delivery file and report."
    )


def launch_audio_mastering_jobs_app(
    *,
    app_title: str = "Definers Mastering Jobs",
    hero_eyebrow: str = "Audio Workflow",
    hero_description: str = (
        "Run mastering as a resumable guided job with persistent artifacts, stem-aware checkpoints, "
        "and a clear next-step summary."
    ),
):
    from html import escape

    import gradio as gr

    from definers.ui.apps.audio_app_services import (
        MASTERING_PROFILE_CHOICES,
        STEM_DRUM_EDGE_DEFAULT,
        STEM_GLUE_REVERB_DEFAULT,
        STEM_MODEL_STRATEGY_CHOICES,
        STEM_VOCAL_PULLBACK_DB_DEFAULT,
        build_mastering_job_mix,
        describe_stem_model_choice,
        finalize_mastering_job,
        get_mastering_profile_ui_state,
        is_custom_stem_model_strategy,
        prepare_mastering_job,
        refresh_mastering_job,
        render_mastering_job_view,
        separate_mastering_job_stems,
    )
    from definers.ui.apps.audio_workspace import (
        AUDIO_FORMAT_CHOICES,
        prepare_audio_workspace,
    )
    from definers.ui.gradio_shared import (
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
        init_status_card,
        launch_blocks,
    )

    prepare_audio_workspace()

    initial_mastering_state = get_mastering_profile_ui_state(
        MASTERING_PROFILE_CHOICES[0]
    )
    initial_stem_strategy_note = describe_stem_model_choice(
        STEM_MODEL_STRATEGY_CHOICES[0]
    )

    with gr.Blocks(title=app_title) as app:
        gr.HTML(
            f"""<div class=\"audio-hero\"><p class=\"eyebrow\">{escape(hero_eyebrow)}</p><h1>{escape(app_title)}</h1><p>{escape(hero_description)}</p></div>"""
        )
        progress_status = init_progress_tracker(
            "Guided mastering ready",
            "Prepare a job to begin. The job folder keeps the workflow resumable.",
        )
        init_output_folder_controls(section="audio/mastering_jobs")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload Mix", type="filepath")
                format_choice = gr.Radio(
                    AUDIO_FORMAT_CHOICES,
                    label="Output Format",
                    value="WAV",
                )
                profile_name = gr.Dropdown(
                    MASTERING_PROFILE_CHOICES,
                    label="Mastering Strategy",
                    value=str(initial_mastering_state["label"]),
                )
                profile_note = gr.Markdown(
                    value=str(initial_mastering_state["description"])
                )
                stem_mastering = gr.Checkbox(
                    label="Use stem-aware mastering",
                    value=False,
                )
                with gr.Accordion("Macro Controls", open=False):
                    macro_note = gr.Markdown(
                        value=str(initial_mastering_state["macro_note"])
                    )
                    bass = gr.Slider(
                        0.0,
                        1.0,
                        float(initial_mastering_state["bass"]),
                        step=0.05,
                        label="Bass",
                        interactive=bool(
                            initial_mastering_state["controls_enabled"]
                        ),
                    )
                    volume = gr.Slider(
                        0.0,
                        1.0,
                        float(initial_mastering_state["volume"]),
                        step=0.05,
                        label="Volume",
                        interactive=bool(
                            initial_mastering_state["controls_enabled"]
                        ),
                    )
                    effects = gr.Slider(
                        0.0,
                        1.0,
                        float(initial_mastering_state["effects"]),
                        step=0.05,
                        label="Effects",
                        interactive=bool(
                            initial_mastering_state["controls_enabled"]
                        ),
                    )
                with gr.Accordion("Stem-Aware Path", open=False):
                    with gr.Group(visible=False) as stem_settings:
                        stem_model_name = gr.Dropdown(
                            STEM_MODEL_STRATEGY_CHOICES,
                            label="Stem Separation Strategy",
                            value=STEM_MODEL_STRATEGY_CHOICES[0],
                        )
                        stem_model_override = gr.Textbox(
                            label="Custom separator checkpoint",
                            placeholder="Example: htdemucs_6s or custom_model.yaml",
                            visible=False,
                        )
                        stem_strategy_note = gr.Markdown(
                            value=initial_stem_strategy_note
                        )
                        stem_shifts = gr.Slider(
                            1,
                            8,
                            2,
                            step=1,
                            label="Stem Separation Shifts",
                        )
                        stem_mix_headroom_db = gr.Slider(
                            3.0,
                            12.0,
                            6.0,
                            step=0.5,
                            label="Stem Mix Headroom (dB)",
                        )
                        save_mastered_stems = gr.Checkbox(
                            label="Save mastered stems",
                            value=True,
                        )
                        stem_glue_reverb_amount = gr.Slider(
                            0.0,
                            1.5,
                            STEM_GLUE_REVERB_DEFAULT,
                            step=0.05,
                            label="Vocal/Other Glue Reverb",
                        )
                        stem_drum_edge_amount = gr.Slider(
                            0.0,
                            1.5,
                            STEM_DRUM_EDGE_DEFAULT,
                            step=0.05,
                            label="Drum Edge Amount",
                        )
                        stem_vocal_pullback_db = gr.Slider(
                            0.0,
                            3.0,
                            STEM_VOCAL_PULLBACK_DB_DEFAULT,
                            step=0.1,
                            label="Extra Vocal Pullback (dB)",
                        )
                compute_note = gr.Markdown(value=_mastering_compute_note(False))
            with gr.Column(scale=1):
                status_card = init_status_card(
                    "Guided mastering ready",
                    "Prepare a job to begin. The job folder keeps the workflow resumable.",
                )
                job_dir = gr.Textbox(
                    label="Job Folder",
                    placeholder="Filled after Prepare Job or paste an existing job folder to resume.",
                    interactive=True,
                )
                with gr.Row():
                    prepare_button = gr.Button(
                        "1. Prepare Job",
                        variant="primary",
                    )
                    separate_button = gr.Button("2. Separate Stems")
                    mix_button = gr.Button("3. Build Stem Mix")
                    finalize_button = gr.Button(
                        "4. Finalize Master",
                        variant="primary",
                    )
                    refresh_button = gr.Button("Refresh Job")

        with gr.Row():
            raw_stems = gr.File(label="Raw Stems", file_count="multiple")
            processed_stems = gr.File(
                label="Processed Stems",
                file_count="multiple",
            )

        with gr.Row():
            mixed_audio = gr.Audio(
                label="Stem Mix Preview",
                interactive=False,
                buttons=["download"],
            )
            mastered_audio = gr.Audio(
                label="Final Master",
                interactive=False,
                buttons=["download"],
            )

        report_file = gr.File(label="Mastering Report", interactive=False)
        report_summary = gr.Markdown()
        with gr.Accordion("Advanced Job Details", open=False):
            manifest_view = gr.Markdown()

        shared_outputs = [
            job_dir,
            status_card,
            raw_stems,
            processed_stems,
            mixed_audio,
            mastered_audio,
            report_file,
            report_summary,
            manifest_view,
        ]

        def prepare_job_view(
            audio_path,
            output_format,
            requested_profile,
            bass_value,
            volume_value,
            effects_value,
            stem_mastering_enabled,
            selected_stem_model_name,
            stem_shifts_value,
            stem_mix_headroom_value,
            save_mastered_stems_value,
            selected_stem_model_override,
            stem_glue_reverb_amount_value,
            stem_drum_edge_amount_value,
            stem_vocal_pullback_db_value,
        ):
            manifest = prepare_mastering_job(
                audio_path,
                output_format,
                requested_profile,
                bass_value,
                volume_value,
                effects_value,
                stem_mastering_enabled,
                selected_stem_model_name,
                stem_shifts_value,
                stem_mix_headroom_value,
                save_mastered_stems_value,
                stem_model_override=selected_stem_model_override,
                stem_glue_reverb_amount=stem_glue_reverb_amount_value,
                stem_drum_edge_amount=stem_drum_edge_amount_value,
                stem_vocal_pullback_db=stem_vocal_pullback_db_value,
            )
            return render_mastering_job_view(
                str(manifest["job_dir"]),
                title="Job prepared",
            )

        def separate_job_view(current_job_dir):
            manifest = separate_mastering_job_stems(current_job_dir)
            if not bool(
                dict(manifest.get("settings", {})).get("stem_mastering")
            ):
                return render_mastering_job_view(
                    current_job_dir,
                    title="Stereo-only job",
                    detail="Stem separation is disabled for this job. Go directly to Finalize Master.",
                )
            return render_mastering_job_view(current_job_dir)

        def mix_job_view(current_job_dir):
            manifest = build_mastering_job_mix(current_job_dir)
            if not bool(
                dict(manifest.get("settings", {})).get("stem_mastering")
            ):
                return render_mastering_job_view(
                    current_job_dir,
                    title="Stereo-only job",
                    detail="No stem mix is needed for this job. Go directly to Finalize Master.",
                )
            return render_mastering_job_view(current_job_dir)

        def finalize_job_view(current_job_dir):
            finalize_mastering_job(current_job_dir)
            return render_mastering_job_view(current_job_dir)

        def refresh_job_view(current_job_dir):
            refresh_mastering_job(current_job_dir)
            return render_mastering_job_view(
                current_job_dir,
                title="Job loaded",
                detail="Resume from the next unfinished step.",
            )

        def update_mastering_profile_ui(
            requested_profile,
            bass_value,
            volume_value,
            effects_value,
        ):
            state = get_mastering_profile_ui_state(
                requested_profile,
                bass_value,
                volume_value,
                effects_value,
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
            selected_model_name,
            selected_model_override,
        ):
            enabled = bool(stem_mastering_enabled)
            strategy_note = (
                describe_stem_model_choice(
                    selected_model_name,
                    selected_model_override,
                )
                if enabled
                else "**Stem Strategy:** Stem-aware mastering is off. The job stays on the stereo-only path."
            )
            return (
                gr.update(visible=enabled),
                gr.update(
                    visible=enabled
                    and is_custom_stem_model_strategy(selected_model_name)
                ),
                gr.update(value=str(strategy_note)),
                gr.update(value=_mastering_compute_note(enabled)),
            )

        profile_name.change(
            update_mastering_profile_ui,
            [profile_name, bass, volume, effects],
            [bass, volume, effects, profile_note, macro_note],
        )

        stem_mastering.change(
            update_mastering_stem_ui,
            [stem_mastering, stem_model_name, stem_model_override],
            [
                stem_settings,
                stem_model_override,
                stem_strategy_note,
                compute_note,
            ],
        )
        stem_model_name.change(
            update_mastering_stem_ui,
            [stem_mastering, stem_model_name, stem_model_override],
            [
                stem_settings,
                stem_model_override,
                stem_strategy_note,
                compute_note,
            ],
        )
        stem_model_override.change(
            update_mastering_stem_ui,
            [stem_mastering, stem_model_name, stem_model_override],
            [
                stem_settings,
                stem_model_override,
                stem_strategy_note,
                compute_note,
            ],
        )

        bind_progress_click(
            prepare_button,
            prepare_job_view,
            progress_output=progress_status,
            inputs=[
                audio_input,
                format_choice,
                profile_name,
                bass,
                volume,
                effects,
                stem_mastering,
                stem_model_name,
                stem_shifts,
                stem_mix_headroom_db,
                save_mastered_stems,
                stem_model_override,
                stem_glue_reverb_amount,
                stem_drum_edge_amount,
                stem_vocal_pullback_db,
            ],
            outputs=shared_outputs,
            action_label="Prepare Job",
            steps=(
                "Validate source",
                "Analyze mastering input",
                "Write job manifest",
                "Publish job",
            ),
            running_detail="Preparing the guided mastering job.",
            success_detail="Mastering job is ready for the next step.",
        )
        bind_progress_click(
            separate_button,
            separate_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=shared_outputs,
            action_label="Separate Stems",
            steps=(
                "Load job",
                "Separate stems",
                "Publish raw stems",
            ),
            running_detail="Running the stem-separation stage.",
            success_detail="Stem separation step is complete.",
        )
        bind_progress_click(
            mix_button,
            mix_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=shared_outputs,
            action_label="Build Stem Mix",
            steps=(
                "Load job",
                "Process separated stems",
                "Publish mix artifacts",
            ),
            running_detail="Building the guided stem mix.",
            success_detail="Stem mix step is complete.",
        )
        bind_progress_click(
            finalize_button,
            finalize_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=shared_outputs,
            action_label="Finalize Master",
            steps=(
                "Load job",
                "Render final delivery",
                "Publish master",
            ),
            running_detail="Rendering the final master and report.",
            success_detail="Final master is ready.",
        )
        bind_progress_click(
            refresh_button,
            refresh_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=shared_outputs,
            action_label="Refresh Job",
            steps=(
                "Load job",
                "Refresh artifacts",
                "Publish status",
            ),
            running_detail="Refreshing the guided mastering job.",
            success_detail="Job state is refreshed.",
        )

    launch_blocks(app)


__all__ = ("launch_audio_mastering_jobs_app",)
