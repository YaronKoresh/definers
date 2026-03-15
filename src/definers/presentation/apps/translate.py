import definers.text as text
from definers.audio import value_to_keys
from definers.constants import language_codes
from definers.ml import init_pretrained_model
from definers.presentation.gradio_shared import launch_blocks
from definers.system import unique


def launch_translate_app():
    import gradio as gr

    init_pretrained_model("translate", True)

    def handle_translate(txt, tgt_lang):
        return text.ai_translate(
            txt,
            value_to_keys(language_codes, tgt_lang)[0],
        )

    with gr.Blocks() as app:
        gr.Markdown("# AI Translator")
        gr.Markdown("### An AI-based translation software for the community")
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(
                    placeholder="Some text...",
                    value="",
                    lines=4,
                    label="Input",
                    container=True,
                    max_length=2000,
                )
                lang = gr.Dropdown(
                    choices=unique(language_codes.values()),
                    value="english",
                )
            with gr.Column():
                res = gr.Textbox(
                    label="Results",
                    container=True,
                    value="",
                    lines=6,
                    show_copy_button=True,
                )
        btn = gr.Button(value="Translate")
        btn.click(fn=handle_translate, inputs=[txt, lang], outputs=[res])
    launch_blocks(app)
