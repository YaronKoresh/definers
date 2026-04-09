class ImageApp:
    @staticmethod
    def title_image(image_path, top, middle, bottom):
        from definers.image import write_on_image

        return write_on_image(image_path, top, middle, bottom)

    @staticmethod
    def upscale_image(path):
        from definers.image import upscale

        return upscale(path)

    @staticmethod
    def generate_image(text, width, height):
        from definers.image import get_max_resolution
        from definers.ml import optimize_prompt_realism, pipe
        from definers.text.validation import TextInputValidator

        validator = TextInputValidator.default()
        validated_text = validator.validate(text)
        width, height = get_max_resolution(width, height, mega_pixels=2.5)
        validated_text = optimize_prompt_realism(validated_text)
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
        from definers.ui.gradio_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
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
            init_output_folder_controls()
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
