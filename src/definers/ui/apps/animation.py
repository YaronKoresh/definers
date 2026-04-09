class AnimationApp:
    @staticmethod
    def generate_chunk(txt, img, dur, seed, chunk_state, _progress=None):
        import math
        import random

        import gradio as gr
        import torch
        from diffusers.utils import export_to_gif
        from PIL import Image, ImageOps

        from definers.constants import MODELS
        from definers.cuda import device
        from definers.ml import optimize_prompt_realism
        from definers.system import full_path
        from definers.text.validation import TextInputValidator

        validator = TextInputValidator.default()
        frames_per_chunk = 5
        fps = 20
        steps = 30
        validated_text = validator.validate(txt)
        validated_text = optimize_prompt_realism(validated_text)
        total_frames = int(dur * fps)
        total_chunks = math.ceil(total_frames / frames_per_chunk)
        current_chunk_index = chunk_state["current_chunk"]
        if current_chunk_index > total_chunks:
            raise gr.Error(
                "All chunks have been generated. Please combine them now."
            )
        if current_chunk_index == 1:
            input_image = ImageOps.fit(
                img,
                (1024, 576),
                Image.Resampling.LANCZOS,
            )
            gr.Info("Generating first chunk using the initial image...")
        else:
            previous_chunk_path = chunk_state["chunk_paths"][-1]
            with Image.open(previous_chunk_path) as gif:
                gif.seek(gif.n_frames - 1)
                input_image = gif.copy()
            gr.Info(
                f"Generating chunk {current_chunk_index}/{total_chunks} using context from previous chunk..."
            )
        if input_image.mode == "RGBA":
            input_image = input_image.convert("RGB")
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device()).manual_seed(
            int(seed) + current_chunk_index
        )
        output = MODELS["video"](
            prompt=validated_text,
            image=input_image,
            generator=generator,
            num_inference_steps=steps,
            num_frames=frames_per_chunk,
        )
        chunk_path = full_path(
            chunk_state["chunks_path"],
            f"chunk_{current_chunk_index}.gif",
        )
        export_to_gif(output.frames[0], chunk_path, fps=fps)
        chunk_state["chunk_paths"].append(chunk_path)
        chunk_state["current_chunk"] += 1
        if current_chunk_index == total_chunks:
            pass
        return (
            chunk_path,
            chunk_state,
            gr.update(visible=True),
        )

    @staticmethod
    def combine_chunks(chunk_state):
        from pathlib import Path

        import gradio as gr
        from PIL import Image

        fps = 20
        if not chunk_state["chunk_paths"]:
            raise RuntimeError("No chunks to combine.")
        all_frames = []
        for chunk_path in chunk_state["chunk_paths"]:
            with Image.open(chunk_path) as gif:
                for index in range(gif.n_frames):
                    gif.seek(index)
                    all_frames.append(gif.copy().convert("RGBA"))
        final_gif_path = str(
            Path(chunk_state["chunks_path"]) / "final_animation.gif"
        )
        all_frames[0].save(
            final_gif_path,
            save_all=True,
            append_images=all_frames[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=True,
        )
        return final_gif_path, gr.update(visible=False)

    @staticmethod
    def reset_state(chunk_state):
        import gradio as gr

        from definers.system.output_paths import managed_output_session_dir

        chunk_state["current_chunk"] = 1
        chunk_state["chunk_paths"] = []
        chunk_state["chunks_path"] = managed_output_session_dir(
            "animation",
            stem="chunks",
        )
        return (
            chunk_state,
            None,
            gr.update(visible=False),
            gr.update(interactive=True),
        )

    @staticmethod
    def launch_animation_app():
        import gradio as gr

        from definers.constants import MAX_INPUT_LENGTH
        from definers.system.output_paths import managed_output_session_dir
        from definers.ui.gradio_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
        )

        with gr.Blocks() as app:
            chunk_state = gr.State(
                {
                    "current_chunk": 1,
                    "chunk_paths": [],
                    "chunks_path": managed_output_session_dir(
                        "animation",
                        stem="chunks",
                    ),
                }
            )
            gr.Markdown("# Image to Animation: Manual Chunking Method")
            gr.Markdown(
                "This app generates long animations piece-by-piece to avoid timeouts on free services. **You must click 'Generate Next Chunk' repeatedly** until all chunks are created, then click 'Combine'."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Input Image", type="pil", height=420)
                    txt = gr.Textbox(
                        placeholder="Describe the scene",
                        value="",
                        lines=4,
                        label="Prompt",
                        container=True,
                        max_length=MAX_INPUT_LENGTH,
                    )
                    dur = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=3,
                        step=1,
                        label="Total Duration (s)",
                    )
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(
                            label="Seed (-1 for random)",
                            minimum=-1,
                            value=-1,
                        )
                with gr.Column(scale=1):
                    out = gr.Image(
                        label="Latest Generated Chunk / Final Animation",
                        interactive=False,
                        height=420,
                    )
                    prog = init_progress_tracker(
                        "Animation ready",
                        "Ready to generate the first chunk.",
                    )
                    init_output_folder_controls()
            with gr.Row():
                generate_button = gr.Button(
                    "Generate Next Chunk",
                    variant="primary",
                )
                combine_button = gr.Button(
                    "Combine Chunks into Final GIF",
                    variant="stop",
                    visible=False,
                )
                reset_button = gr.Button("Start Over")
            bind_progress_click(
                generate_button,
                AnimationApp.generate_chunk,
                progress_output=prog,
                inputs=[txt, img, dur, seed, chunk_state],
                outputs=[out, chunk_state, combine_button],
                action_label="Generate Next Chunk",
                steps=(
                    "Validate inputs",
                    "Render next chunk",
                    "Publish chunk",
                ),
                running_detail="Generating the next animation chunk.",
                success_detail="Chunk generation finished.",
            )
            bind_progress_click(
                combine_button,
                AnimationApp.combine_chunks,
                progress_output=prog,
                inputs=[chunk_state],
                outputs=[out, combine_button],
                action_label="Combine Chunks",
                steps=(
                    "Validate chunk set",
                    "Combine animation frames",
                    "Publish animation",
                ),
                running_detail="Combining the generated chunks.",
                success_detail="Final animation is ready.",
            )
            bind_progress_click(
                reset_button,
                AnimationApp.reset_state,
                progress_output=prog,
                inputs=[chunk_state],
                outputs=[
                    chunk_state,
                    out,
                    combine_button,
                    generate_button,
                ],
                action_label="Reset Animation Session",
                steps=(
                    "Clear session",
                    "Rebuild chunk workspace",
                    "Reset controls",
                ),
                running_detail="Resetting the chunk session.",
                success_detail="Animation session has been reset.",
            )
        launch_blocks(app)


launch_animation_app = AnimationApp.launch_animation_app
