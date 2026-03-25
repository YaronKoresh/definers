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
        from definers.application_text import TextInputValidator
        from definers.image import get_max_resolution
        from definers.ml import optimize_prompt_realism, pipe

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
    def launch_image_app():
        import gradio as gr

        from definers.constants import MAX_INPUT_LENGTH
        from definers.image import init_upscale
        from definers.ml import init_pretrained_model
        from definers.presentation.gradio_shared import launch_blocks

        init_pretrained_model("translate", True)
        init_pretrained_model("summary", True)
        init_pretrained_model("image", True)
        init_upscale()

        with gr.Blocks() as app:
            gr.Markdown("# Text-to-Image generator")
            gr.Markdown("### Realistic. Upscalable. Multilingual.")
            with gr.Row():
                with gr.Column(scale=1):
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
                        interactive=False,
                        label="Result",
                        type="filepath",
                        show_share_button=False,
                        container=True,
                        show_download_button=True,
                    )
                    generate_image = gr.Button("Generate")
                    upscale_now = gr.Button("Upscale")
                    add_titles = gr.Button("Add title(s)")
            generate_image.click(
                fn=ImageApp.generate_image,
                inputs=[data, width_input, height_input],
                outputs=[cover],
            )
            upscale_now.click(
                fn=ImageApp.upscale_image,
                inputs=[cover],
                outputs=[cover],
            )
            add_titles.click(
                fn=ImageApp.title_image,
                inputs=[cover, top, middle, bottom],
                outputs=[cover],
            )
        launch_blocks(app)


launch_image_app = ImageApp.launch_image_app
