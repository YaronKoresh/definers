from __future__ import annotations


def generate_chunk(txt, img, dur, seed, chunk_state, _progress=None):
    import math
    import random

    import torch
    from diffusers.utils import export_to_gif
    from PIL import Image, ImageOps

    from definers.constants import MODELS
    from definers.cuda import device
    from definers.ml import optimize_prompt_realism
    from definers.system import full_path
    from definers.system.download_activity import create_activity_reporter
    from definers.text.validation import TextInputValidator

    validator = TextInputValidator.default()
    frames_per_chunk = 5
    fps = 20
    steps = 30
    report = create_activity_reporter(5)
    report(
        1,
        "Validate animation prompt",
        detail="Checking the prompt and chunk settings.",
    )
    validated_text = validator.validate(txt)
    report(
        2,
        "Prepare prompt",
        detail="Optimizing the prompt for animation generation.",
    )
    validated_text = optimize_prompt_realism(validated_text)
    total_frames = int(dur * fps)
    total_chunks = math.ceil(total_frames / frames_per_chunk)
    current_chunk_index = chunk_state["current_chunk"]
    if current_chunk_index > total_chunks:
        raise RuntimeError(
            "All chunks have been generated. Please combine them now."
        )
    report(
        3,
        "Prepare source frame",
        detail=f"Preparing chunk {current_chunk_index}/{total_chunks}.",
    )
    if current_chunk_index == 1:
        source_image = Image.open(img) if isinstance(img, str) else img
        input_image = ImageOps.fit(
            source_image,
            (1024, 576),
            Image.Resampling.LANCZOS,
        )
        print("Generating first chunk using the initial image...")
    else:
        previous_chunk_path = chunk_state["chunk_paths"][-1]
        with Image.open(previous_chunk_path) as gif:
            gif.seek(gif.n_frames - 1)
            input_image = gif.copy()
        print(
            f"Generating chunk {current_chunk_index}/{total_chunks}"
            " using context from previous chunk..."
        )
    if input_image.mode == "RGBA":
        input_image = input_image.convert("RGB")
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device()).manual_seed(
        int(seed) + current_chunk_index
    )
    report(
        4,
        "Render chunk",
        detail=f"Generating chunk {current_chunk_index}/{total_chunks}.",
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
    report(
        5,
        "Write chunk",
        detail=f"Saving chunk {current_chunk_index}/{total_chunks}.",
    )
    export_to_gif(output.frames[0], chunk_path, fps=fps)
    chunk_state["chunk_paths"].append(chunk_path)
    chunk_state["current_chunk"] += 1
    return chunk_path, chunk_state, {"visible": True}


def combine_chunks(chunk_state):
    from pathlib import Path

    from PIL import Image

    from definers.system.download_activity import create_activity_reporter

    fps = 20
    chunk_paths = list(chunk_state["chunk_paths"])
    if not chunk_paths:
        raise RuntimeError("No chunks to combine.")
    report = create_activity_reporter(len(chunk_paths) + 2)
    report(
        1,
        "Validate chunk set",
        detail="Checking the generated animation chunks.",
    )
    all_frames = []
    for chunk_index, chunk_path in enumerate(chunk_paths, start=1):
        report(
            chunk_index + 1,
            "Collect chunk frames",
            detail=f"Loading frames from chunk {chunk_index}/{len(chunk_paths)}.",
        )
        with Image.open(chunk_path) as gif:
            for frame_index in range(gif.n_frames):
                gif.seek(frame_index)
                all_frames.append(gif.copy().convert("RGBA"))
    final_gif_path = str(
        Path(chunk_state["chunks_path"]) / "final_animation.gif"
    )
    report(
        len(chunk_paths) + 2,
        "Write final animation",
        detail="Saving the combined animation GIF.",
    )
    all_frames[0].save(
        final_gif_path,
        save_all=True,
        append_images=all_frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
    )
    return final_gif_path, {"visible": False}


def reset_state(chunk_state):
    from definers.system.download_activity import create_activity_reporter
    from definers.system.output_paths import (
        cleanup_managed_output_path,
        managed_output_session_dir,
    )

    report = create_activity_reporter(3)
    report(
        1,
        "Clear chunk session",
        detail="Clearing the current animation chunk state.",
    )
    cleanup_managed_output_path(chunk_state.get("chunks_path"))
    chunk_state["current_chunk"] = 1
    chunk_state["chunk_paths"] = []
    report(
        2,
        "Create chunk workspace",
        detail="Preparing a fresh animation output workspace.",
    )
    chunk_state["chunks_path"] = managed_output_session_dir(
        "animation",
        stem="chunks",
    )
    report(
        3,
        "Reset animation controls",
        detail="Resetting the animation controls to the initial state.",
    )
    return (
        chunk_state,
        None,
        {"visible": False},
        {"interactive": True},
    )


def launch_animation_app():
    from PySide6.QtWidgets import QPushButton

    from definers.constants import MAX_INPUT_LENGTH
    from definers.system.output_paths import managed_output_session_dir
    from definers.ui.gui_shared import (
        State,
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
        launch_blocks,
        qt_app,
    )
    from definers.ui.qt_widgets import (
        Accordion,
        ImageInput,
        ImageOutput,
        LabeledSlider,
        LabeledSpin,
        LabeledTextbox,
        MarkdownLabel,
        Row,
        Section,
        build_main_window,
    )

    qt_app()

    chunk_state = State(
        {
            "current_chunk": 1,
            "chunk_paths": [],
            "chunks_path": managed_output_session_dir(
                "animation",
                stem="chunks",
            ),
        }
    )

    body = Section()
    body.addWidget(
        MarkdownLabel("# Image to Animation: Manual Chunking Method")
    )
    body.addWidget(
        MarkdownLabel(
            "This app generates long animations piece-by-piece to avoid"
            " timeouts on free services. **You must click 'Generate Next"
            " Chunk' repeatedly** until all chunks are created, then click"
            " 'Combine'."
        )
    )

    columns = Row()
    left = Section()
    img = ImageInput("Input Image", height=420)
    txt = LabeledTextbox(
        "Prompt",
        placeholder="Describe the scene",
        lines=4,
        max_length=MAX_INPUT_LENGTH,
    )
    dur = LabeledSlider(
        "Total Duration (s)",
        minimum=1,
        maximum=30,
        value=3,
        step=1,
    )
    advanced = Accordion("Advanced Settings", open=False)
    seed = LabeledSpin(
        "Seed (-1 for random)",
        minimum=-1,
        maximum=2_000_000_000,
        value=-1,
    )
    advanced.addWidget(seed)
    left.addWidget(img)
    left.addWidget(txt)
    left.addWidget(dur)
    left.addWidget(advanced)

    right = Section()
    out = ImageOutput(
        "Latest Generated Chunk / Final Animation",
        height=420,
    )
    prog = init_progress_tracker(
        "Animation ready",
        "Ready to generate the first chunk.",
    )
    _, output_toolbar = init_output_folder_controls(section="animation")
    right.addWidget(out)
    right.addWidget(prog)
    right.addWidget(output_toolbar)

    columns.addWidget(left)
    columns.addWidget(right)
    body.addWidget(columns)

    button_row = Row()
    generate_button = QPushButton("Generate Next Chunk")
    generate_button.setProperty("definersPrimary", True)
    combine_button = QPushButton("Combine Chunks into Final GIF")
    combine_button.setVisible(False)
    reset_button = QPushButton("Start Over")
    button_row.addWidget(generate_button)
    button_row.addWidget(combine_button)
    button_row.addWidget(reset_button)
    body.addWidget(button_row)

    bind_progress_click(
        generate_button,
        generate_chunk,
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
        combine_chunks,
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
        reset_state,
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

    window = build_main_window("Definers Animation Studio", body)
    launch_blocks(window)
