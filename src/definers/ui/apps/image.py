from __future__ import annotations


def title_image(image_path, top, middle, bottom):
    from definers.image import write_on_image
    from definers.system.download_activity import create_activity_reporter

    report = create_activity_reporter(3)
    report(
        1,
        "Validate source image",
        detail="Checking the selected image and title inputs.",
    )
    report(
        2,
        "Render title overlays",
        detail="Rendering the requested title overlays.",
    )
    result = write_on_image(image_path, top, middle, bottom)
    report(
        3,
        "Finalize titled image",
        detail="Saving the titled image output.",
    )
    return result


def upscale_image(path):
    from definers.image import upscale
    from definers.system.download_activity import create_activity_reporter

    report = create_activity_reporter(3)
    report(
        1,
        "Validate source image",
        detail="Checking the selected image for upscaling.",
    )
    report(
        2,
        "Prepare upscaler",
        detail="Loading the upscaling runtime and checkpoints.",
    )
    result = upscale(path)
    report(
        3,
        "Finalize upscale",
        detail="Saving the upscaled image output.",
    )
    return result


def generate_image(text, width, height):
    from definers.image import get_max_resolution
    from definers.ml import optimize_prompt_realism, pipe
    from definers.system.download_activity import create_activity_reporter
    from definers.text.validation import TextInputValidator

    report = create_activity_reporter(4)
    report(
        1,
        "Validate prompt",
        detail="Checking the image prompt.",
    )
    validator = TextInputValidator.default()
    validated_text = validator.validate(text)
    report(
        2,
        "Resolve canvas",
        detail="Calculating the target image resolution.",
    )
    width, height = get_max_resolution(width, height, mega_pixels=2.5)
    report(
        3,
        "Prepare prompt",
        detail="Optimizing the prompt for image generation.",
    )
    validated_text = optimize_prompt_realism(validated_text)
    report(
        4,
        "Generate image",
        detail="Running the image generation model.",
    )
    return pipe(
        "image",
        prompt=validated_text,
        resolution=f"{width}x{height}",
    )


def prepare_image_job_view(
    prompt_value,
    width_value,
    height_value,
    top_value,
    middle_value,
    bottom_value,
):
    from definers.ui.apps.image_generate_jobs import (
        prepare_image_generate_job,
        render_image_job_view,
    )

    manifest = prepare_image_generate_job(
        prompt_value,
        width_value,
        height_value,
        top_value,
        middle_value,
        bottom_value,
    )
    return render_image_job_view(str(manifest["job_dir"]), title="Job prepared")


def run_full_image_job_view(
    prompt_value,
    width_value,
    height_value,
    top_value,
    middle_value,
    bottom_value,
):
    from definers.ui.apps.image_generate_jobs import (
        render_image_job_view,
        run_full_image_generate_job,
    )

    manifest = run_full_image_generate_job(
        prompt_value,
        width_value,
        height_value,
        top_value,
        middle_value,
        bottom_value,
    )
    return render_image_job_view(
        str(manifest["job_dir"]),
        title="Job completed",
        detail="The full staged image flow finished and published the artifacts.",
    )


def generate_image_job_view(current_job_dir):
    from definers.ui.apps.image_generate_jobs import (
        generate_image_job,
        render_image_job_view,
    )

    generate_image_job(current_job_dir)
    return render_image_job_view(current_job_dir)


def upscale_image_job_view(current_job_dir):
    from definers.ui.apps.image_generate_jobs import (
        render_image_job_view,
        upscale_image_job,
    )

    upscale_image_job(current_job_dir)
    return render_image_job_view(current_job_dir)


def title_image_job_view(
    current_job_dir,
    top_value,
    middle_value,
    bottom_value,
):
    from definers.ui.apps.image_generate_jobs import (
        render_image_job_view,
        title_image_job,
    )

    title_image_job(
        current_job_dir,
        top_value,
        middle_value,
        bottom_value,
    )
    return render_image_job_view(current_job_dir)


def refresh_image_job_view(current_job_dir):
    from definers.ui.apps.image_generate_jobs import (
        refresh_image_generate_job,
        render_image_job_view,
    )

    refresh_image_generate_job(current_job_dir)
    return render_image_job_view(
        current_job_dir,
        title="Job loaded",
        detail="Resume from the next unfinished step.",
    )


def launch_image_app(
    steps=None,
    *,
    app_title="Definers Image Studio",
    hero_eyebrow="Definers Image",
    hero_description=(
        "Generate a new image, upscale an existing result, or add simple"
        " titles without leaving the surface."
    ),
):
    from html import escape

    from PySide6.QtWidgets import QPushButton

    from definers.constants import MAX_INPUT_LENGTH
    from definers.ui.gui_shared import (
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
        init_status_card,
        launch_blocks,
        qt_app,
    )
    from definers.ui.qt_widgets import (
        Accordion,
        HeroCard,
        ImageInput,
        ImageOutput,
        LabeledLineEdit,
        LabeledSlider,
        LabeledTextbox,
        MarkdownLabel,
        Row,
        Section,
        build_main_window,
    )

    qt_app()

    enabled_steps = set(steps or ("generate", "upscale", "title"))

    body = Section()
    body.addWidget(
        HeroCard(
            f"<div>"
            f'<p style="color:#bdf6ff;font-size:11px;font-weight:700;'
            f'letter-spacing:3px;margin:0 0 10px 0;">'
            f"{escape(hero_eyebrow)}</p>"
            f'<div style="color:#f9fcff;font-size:24px;font-weight:700;'
            f'margin:0 0 8px 0;">{escape(app_title)}</div>'
            f'<p style="color:#c8d3e2;font-size:14px;line-height:1.55;'
            f'margin:0;">{escape(hero_description)}</p>'
            f"</div>"
        )
    )
    progress_status = init_progress_tracker(
        "Image studio ready",
        "Choose an image action and run it.",
    )
    body.addWidget(progress_status)
    _, output_toolbar = init_output_folder_controls(section="image")
    body.addWidget(output_toolbar)

    main_row = Row()
    left = Section()
    width_input = None
    height_input = None
    prompt_input = None
    top = None
    middle = None
    bottom = None

    if "generate" in enabled_steps:
        width_input = LabeledSlider(
            "Width",
            minimum=1,
            maximum=16,
            step=1,
            value=4,
        )
        height_input = LabeledSlider(
            "Height",
            minimum=1,
            maximum=16,
            step=1,
            value=4,
        )
        prompt_input = LabeledTextbox(
            "Prompt",
            placeholder="Input data",
            lines=4,
            max_length=MAX_INPUT_LENGTH,
        )
        left.addWidget(width_input)
        left.addWidget(height_input)
        left.addWidget(prompt_input)

    if "title" in enabled_steps:
        top = LabeledLineEdit("Top Title", placeholder="Top title")
        middle = LabeledLineEdit("Middle Title", placeholder="Middle title")
        bottom = LabeledLineEdit("Bottom Title", placeholder="Bottom title")
        left.addWidget(top)
        left.addWidget(middle)
        left.addWidget(bottom)

    right = Section()
    cover = ImageInput("Image", height=360)
    right.addWidget(cover)
    generate_button = upscale_button = title_button = None
    if "generate" in enabled_steps:
        generate_button = QPushButton("Generate")
        generate_button.setProperty("definersPrimary", True)
        right.addWidget(generate_button)
    if "upscale" in enabled_steps:
        upscale_button = QPushButton("Upscale")
        right.addWidget(upscale_button)
    if "title" in enabled_steps:
        title_button = QPushButton("Add title(s)")
        right.addWidget(title_button)

    main_row.addWidget(left)
    main_row.addWidget(right)
    body.addWidget(main_row)

    if "generate" in enabled_steps:
        stages = Accordion("Run In Stages Or Resume", open=False)
        stages.addWidget(
            MarkdownLabel(
                "Use the current generation settings to prepare a resumable"
                " job, run one stage at a time, or execute the full"
                " generation, upscale, and title flow in one pass."
            )
        )
        job_status = init_status_card(
            "Staged image job ready",
            "Prepare a job to enable resumable generation stages.",
        )
        stages.addWidget(job_status)
        job_dir = LabeledLineEdit(
            "Job Folder",
            placeholder=(
                "Filled after Prepare Staged Job or paste an existing job"
                " folder to resume."
            ),
        )
        stages.addWidget(job_dir)

        titles_box = Accordion("Optional Job Titles", open=False)
        job_top = LabeledLineEdit("Job Top Title")
        job_middle = LabeledLineEdit("Job Middle Title")
        job_bottom = LabeledLineEdit("Job Bottom Title")
        titles_box.addWidget(job_top)
        titles_box.addWidget(job_middle)
        titles_box.addWidget(job_bottom)
        stages.addWidget(titles_box)

        button_row = Row()
        prepare_job_button = QPushButton("Prepare Staged Job")
        run_full_job_button = QPushButton("Run Full Job")
        run_full_job_button.setProperty("definersPrimary", True)
        generate_job_button = QPushButton("Generate Image")
        upscale_job_button = QPushButton("Upscale Result")
        title_job_button = QPushButton("Add Titles")
        refresh_job_button = QPushButton("Refresh Job")
        for button in (
            prepare_job_button,
            run_full_job_button,
            generate_job_button,
            upscale_job_button,
            title_job_button,
            refresh_job_button,
        ):
            button_row.addWidget(button)
        stages.addWidget(button_row)

        artifact_row = Row()
        generated_job_image = ImageOutput("Generated Image", height=220)
        upscaled_job_image = ImageOutput("Upscaled Image", height=220)
        titled_job_image = ImageOutput("Titled Image", height=220)
        artifact_row.addWidget(generated_job_image)
        artifact_row.addWidget(upscaled_job_image)
        artifact_row.addWidget(titled_job_image)
        stages.addWidget(artifact_row)

        manifest_box = Accordion("Advanced Job Details", open=False)
        job_manifest = MarkdownLabel("")
        manifest_box.addWidget(job_manifest)
        stages.addWidget(manifest_box)
        body.addWidget(stages)

        bind_progress_click(
            generate_button,
            generate_image,
            progress_output=progress_status,
            inputs=[prompt_input, width_input, height_input],
            outputs=[cover],
            action_label="Generate Image",
            steps=(
                "Validate prompt",
                "Generate image",
                "Publish result",
            ),
            running_detail="Generating the image.",
            success_detail="Generated image is ready.",
        )

        job_outputs = [
            job_dir,
            job_status,
            generated_job_image,
            upscaled_job_image,
            titled_job_image,
            job_manifest,
        ]
        bind_progress_click(
            prepare_job_button,
            prepare_image_job_view,
            progress_output=progress_status,
            inputs=[
                prompt_input,
                width_input,
                height_input,
                job_top,
                job_middle,
                job_bottom,
            ],
            outputs=job_outputs,
            action_label="Prepare Staged Job",
            steps=(
                "Validate prompt",
                "Write job manifest",
                "Publish job",
            ),
            running_detail="Preparing the resumable image job.",
            success_detail="Staged image job is ready.",
        )
        bind_progress_click(
            run_full_job_button,
            run_full_image_job_view,
            progress_output=progress_status,
            inputs=[
                prompt_input,
                width_input,
                height_input,
                job_top,
                job_middle,
                job_bottom,
            ],
            outputs=job_outputs,
            action_label="Run Full Job",
            steps=(
                "Prepare job",
                "Generate image",
                "Upscale and title",
                "Publish artifacts",
            ),
            running_detail="Running the full staged image flow.",
            success_detail="Full staged image flow is complete.",
        )
        bind_progress_click(
            generate_job_button,
            generate_image_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=job_outputs,
            action_label="Generate Image",
            steps=(
                "Load job",
                "Generate image",
                "Publish artifact",
            ),
            running_detail="Running the staged image generation step.",
            success_detail="Generated image is ready.",
        )
        bind_progress_click(
            upscale_job_button,
            upscale_image_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=job_outputs,
            action_label="Upscale Result",
            steps=(
                "Load job",
                "Upscale image",
                "Publish artifact",
            ),
            running_detail="Running the staged upscale step.",
            success_detail="Upscaled image is ready.",
        )
        bind_progress_click(
            title_job_button,
            title_image_job_view,
            progress_output=progress_status,
            inputs=[job_dir, job_top, job_middle, job_bottom],
            outputs=job_outputs,
            action_label="Add Titles",
            steps=(
                "Load job",
                "Render title overlay",
                "Publish artifact",
            ),
            running_detail="Running the staged title step.",
            success_detail="Titled image is ready.",
        )
        bind_progress_click(
            refresh_job_button,
            refresh_image_job_view,
            progress_output=progress_status,
            inputs=[job_dir],
            outputs=job_outputs,
            action_label="Refresh Job",
            steps=(
                "Load job",
                "Refresh artifacts",
                "Publish status",
            ),
            running_detail="Refreshing the staged image job.",
            success_detail="Staged image job is refreshed.",
        )

    if "upscale" in enabled_steps:
        bind_progress_click(
            upscale_button,
            upscale_image,
            progress_output=progress_status,
            inputs=[cover],
            outputs=[cover],
            action_label="Upscale Image",
            steps=(
                "Validate source image",
                "Upscale image",
                "Publish result",
            ),
            running_detail="Upscaling the selected image.",
            success_detail="Upscaled image is ready.",
        )

    if "title" in enabled_steps:
        bind_progress_click(
            title_button,
            title_image,
            progress_output=progress_status,
            inputs=[cover, top, middle, bottom],
            outputs=[cover],
            action_label="Add Titles",
            steps=(
                "Validate source image",
                "Render title overlays",
                "Publish result",
            ),
            running_detail="Applying title overlays.",
            success_detail="Titled image is ready.",
        )

    window = build_main_window(app_title, body)
    launch_blocks(window)
