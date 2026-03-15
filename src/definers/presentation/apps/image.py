from definers.constants import MAX_INPUT_LENGTH
from definers.image import (
    get_max_resolution,
    init_upscale,
    upscale,
    write_on_image,
)
from definers.ml import init_pretrained_model, optimize_prompt_realism, pipe
from definers.presentation.chat_handlers import validate_text_input
from definers.presentation.gradio_shared import launch_blocks


def launch_image_app():
    import gradio as gr

    init_pretrained_model("translate", True)
    init_pretrained_model("summary", True)
    init_pretrained_model("image", True)
    init_upscale()

    def title(image_path, top, middle, bottom):
        return write_on_image(image_path, top, middle, bottom)

    def handle_generation(text, w, h):
        text = validate_text_input(text)
        w, h = get_max_resolution(w, h, mega_pixels=2.5)
        text = optimize_prompt_realism(text)
        return pipe("image", prompt=text, resolution=f"{w}x{h}")

    def handle_upscaling(path):
        return upscale(path)

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
            fn=handle_generation,
            inputs=[data, width_input, height_input],
            outputs=[cover],
        )
        upscale_now.click(fn=handle_upscaling, inputs=[cover], outputs=[cover])
        add_titles.click(
            fn=title,
            inputs=[cover, top, middle, bottom],
            outputs=[cover],
        )
    launch_blocks(app)
