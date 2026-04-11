class ImageApp:
    @staticmethod
    def title_image(image_path, top, middle, bottom):
        from definers.image import write_on_image
        from definers.system.download_activity import (
            create_activity_reporter,
        )

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

    @staticmethod
    def upscale_image(path):
        from definers.image import upscale
        from definers.system.download_activity import (
            create_activity_reporter,
        )

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

    @staticmethod
    def generate_image(text, width, height):
        from definers.image import get_max_resolution
        from definers.ml import optimize_prompt_realism, pipe
        from definers.system.download_activity import (
            create_activity_reporter,
        )
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

    @staticmethod
    def launch_image_app(
        steps=None,
        *,
        app_title="Definers Image Studio",
        hero_eyebrow="Definers Image",
        hero_description="Generate a new image, upscale an existing result, or add simple titles without leaving the surface.",
    ):
        from html import escape

        import gradio as gr

        from definers.constants import MAX_INPUT_LENGTH
        from definers.ui.apps.image_generate_jobs import (
            generate_image_job,
            prepare_image_generate_job,
            refresh_image_generate_job,
            render_image_job_view,
            run_full_image_generate_job,
            title_image_job,
            upscale_image_job,
        )
        from definers.ui.gradio_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
            init_status_card,
            launch_blocks,
        )

        enabled_steps = set(steps or ("generate", "upscale", "title"))

        with gr.Blocks(title=app_title) as app:
            gr.HTML(
                f"""<div class=\"audio-hero\"><p class=\"eyebrow\">{escape(hero_eyebrow)}</p><h1>{escape(app_title)}</h1><p>{escape(hero_description)}</p></div>"""
            )
            progress_status = init_progress_tracker(
                "Image studio ready",
                "Choose an image action and run it.",
            )
            init_output_folder_controls(section="image")
            with gr.Row():
                with gr.Column(scale=1):
                    if "generate" in enabled_steps:
                        width_input = gr.Slider(
                            minimum=1, maximum=16, step=1, label="Width"
                        )
                        height_input = gr.Slider(
                            minimum=1, maximum=16, step=1, label="Height"
                        )
                        data = gr.Textbox(
                            placeholder="Input data",
                            value="",
                            max_length=MAX_INPUT_LENGTH,
                            lines=4,
                            label="Prompt",
                            container=True,
                        )
                    if "title" in enabled_steps:
                        top = gr.Textbox(
                            placeholder="Top title",
                            value="",
                            max_lines=1,
                            label="Top Title",
                        )
                        middle = gr.Textbox(
                            placeholder="Middle title",
                            value="",
                            max_lines=1,
                            label="Middle Title",
                        )
                        bottom = gr.Textbox(
                            placeholder="Bottom title",
                            value="",
                            max_lines=1,
                            label="Bottom Title",
                        )
                with gr.Column(scale=1):
                    cover = gr.Image(
                        interactive=True,
                        label="Image",
                        type="filepath",
                        buttons=["download"],
                        container=True,
                    )
                    if "generate" in enabled_steps:
                        generate_image = gr.Button("Generate")
                    if "upscale" in enabled_steps:
                        upscale_now = gr.Button("Upscale")
                    if "title" in enabled_steps:
                        add_titles = gr.Button("Add title(s)")
            if "generate" in enabled_steps:
                with gr.Accordion("Run In Stages Or Resume", open=False):
                    gr.Markdown(
                        "Use the current generation settings to prepare a resumable job, run one stage at a time, or execute the full generation, upscale, and title flow in one pass."
                    )
                    job_status = init_status_card(
                        "Staged image job ready",
                        "Prepare a job to enable resumable generation stages.",
                    )
                    job_dir = gr.Textbox(
                        label="Job Folder",
                        placeholder="Filled after Prepare Staged Job or paste an existing job folder to resume.",
                        interactive=True,
                    )
                    with gr.Accordion("Optional Job Titles", open=False):
                        job_top = gr.Textbox(label="Job Top Title")
                        job_middle = gr.Textbox(label="Job Middle Title")
                        job_bottom = gr.Textbox(label="Job Bottom Title")
                    with gr.Row():
                        prepare_job_button = gr.Button("Prepare Staged Job")
                        run_full_job_button = gr.Button(
                            "Run Full Job",
                            variant="primary",
                        )
                        generate_job_button = gr.Button("Generate Image")
                        upscale_job_button = gr.Button("Upscale Result")
                        title_job_button = gr.Button("Add Titles")
                        refresh_job_button = gr.Button("Refresh Job")
                    with gr.Row():
                        generated_job_image = gr.Image(
                            label="Generated Image",
                            interactive=False,
                            type="filepath",
                            buttons=["download"],
                        )
                        upscaled_job_image = gr.Image(
                            label="Upscaled Image",
                            interactive=False,
                            type="filepath",
                            buttons=["download"],
                        )
                        titled_job_image = gr.Image(
                            label="Titled Image",
                            interactive=False,
                            type="filepath",
                            buttons=["download"],
                        )
                    with gr.Accordion("Advanced Job Details", open=False):
                        job_manifest = gr.Markdown()
            if "generate" in enabled_steps:
                bind_progress_click(
                    generate_image,
                    ImageApp.generate_image,
                    progress_output=progress_status,
                    inputs=[data, width_input, height_input],
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

                def prepare_image_job_view(
                    prompt_value,
                    width_value,
                    height_value,
                    top_value,
                    middle_value,
                    bottom_value,
                ):
                    manifest = prepare_image_generate_job(
                        prompt_value,
                        width_value,
                        height_value,
                        top_value,
                        middle_value,
                        bottom_value,
                    )
                    return render_image_job_view(
                        str(manifest["job_dir"]),
                        title="Job prepared",
                    )

                def run_full_image_job_view(
                    prompt_value,
                    width_value,
                    height_value,
                    top_value,
                    middle_value,
                    bottom_value,
                ):
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
                    generate_image_job(current_job_dir)
                    return render_image_job_view(current_job_dir)

                def upscale_image_job_view(current_job_dir):
                    upscale_image_job(current_job_dir)
                    return render_image_job_view(current_job_dir)

                def title_image_job_view(
                    current_job_dir,
                    top_value,
                    middle_value,
                    bottom_value,
                ):
                    title_image_job(
                        current_job_dir,
                        top_value,
                        middle_value,
                        bottom_value,
                    )
                    return render_image_job_view(current_job_dir)

                def refresh_image_job_view(current_job_dir):
                    refresh_image_generate_job(current_job_dir)
                    return render_image_job_view(
                        current_job_dir,
                        title="Job loaded",
                        detail="Resume from the next unfinished step.",
                    )

                bind_progress_click(
                    prepare_job_button,
                    prepare_image_job_view,
                    progress_output=progress_status,
                    inputs=[
                        data,
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
                        data,
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
                    upscale_now,
                    ImageApp.upscale_image,
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
                    add_titles,
                    ImageApp.title_image,
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
        launch_blocks(app)


launch_image_app = ImageApp.launch_image_app
