class VideoApp:
    VIDEO_APP_CSS = """
body { color: #00ff41; font-family: monospace; }
.gr-button.primary { background: #00f3ff; color: black; font-weight: bold; box-shadow: 0 0 10px #00f3ff; }
.gr-button.secondary { background: #222; color: white; border: 1px solid #444; }
.section-header { color: #ff003c; font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px; }
textarea { overflow-y: auto !important; }
"""

    @staticmethod
    def launch_video_app(
        visible_tabs=None,
        *,
        app_title="AI VIDEO ARCHITECT",
        hero_eyebrow="Definers Video",
        hero_description="Compose videos, generate lyric videos, or build a music visualizer from one focused surface.",
    ):
        from html import escape

        import gradio as gr

        from definers.constants import STYLES_DB
        from definers.ui.gradio_shared import (
            bind_progress_click,
            css as shared_css,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
        )
        from definers.ui.lyric_video_service import lyric_video
        from definers.ui.music_video_service import music_video
        from definers.video.gui import filter_styles, generate_video_handler

        video_theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")
        selected_tabs = set(
            visible_tabs or ("composer", "lyrics", "visualizer")
        )

        with gr.Blocks(title=app_title) as app:
            gr.HTML(
                f"""<div class=\"audio-hero\"><p class=\"eyebrow\">{escape(hero_eyebrow)}</p><h1>{escape(app_title)}</h1><p>{escape(hero_description)}</p></div>"""
            )
            progress_status = init_progress_tracker(
                "Video workspace ready",
                "Pick a render workflow and start it.",
            )
            init_output_folder_controls()
            with gr.Tabs():
                if "composer" in selected_tabs:
                    with gr.TabItem("Composer"):
                        gr.Markdown("### Advanced Composition & Layout Engine")
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Group():
                                    gr.Markdown(
                                        "### ðŸ“‚ Media & Style",
                                        elem_classes="section-header",
                                    )
                                    audio_in = gr.Audio(
                                        label="Audio File", type="filepath"
                                    )
                                    with gr.Row():
                                        search_txt = gr.Textbox(
                                            placeholder="Search styles...",
                                            label="Search",
                                            scale=2,
                                        )
                                        cat_filter = gr.Dropdown(
                                            [
                                                "All",
                                                "Abstract",
                                                "Cyberpunk",
                                                "Retro",
                                                "Simple",
                                                "Sci-Fi",
                                            ],
                                            value="All",
                                            label="Category Filter",
                                            scale=1,
                                        )
                                    style_dd = gr.Dropdown(
                                        choices=list(STYLES_DB.keys()),
                                        value="Psychedelic Geometry",
                                        label="Select Style (Base Layer)",
                                    )
                                    search_txt.change(
                                        filter_styles,
                                        [search_txt, cat_filter],
                                        style_dd,
                                    )
                                    cat_filter.change(
                                        filter_styles,
                                        [search_txt, cat_filter],
                                        style_dd,
                                    )

                                    with gr.Row():
                                        with gr.Column():
                                            image_in = gr.Image(
                                                label="Background Image",
                                                type="filepath",
                                            )
                                            resolution = gr.Dropdown(
                                                [
                                                    "Square (1:1)",
                                                    "Portrait (9:16)",
                                                    "Landscape (16:9)",
                                                ],
                                                value="Landscape (16:9)",
                                                label="Resolution",
                                            )
                                            fps = gr.Slider(
                                                minimum=1,
                                                maximum=60,
                                                value=20,
                                                label="FPS",
                                            )
                                            sensitivity = gr.Slider(
                                                minimum=0.1,
                                                maximum=5,
                                                value=1,
                                                label="Sensitivity",
                                            )
                                            reactivity = gr.Dropdown(
                                                ["Low", "Mid", "High", "Full"],
                                                value="Full",
                                                label="Reactivity Band",
                                            )
                                        with gr.Column():
                                            palette = gr.Dropdown(
                                                choices=list(STYLES_DB.keys()),
                                                value=list(STYLES_DB.keys())[0],
                                                label="Color Palette",
                                            )
                                            overlays = gr.CheckboxGroup(
                                                [
                                                    "Neon Border",
                                                    "Progress Bar",
                                                    "Audio Waveform",
                                                    "Timer",
                                                ],
                                                label="Overlays",
                                            )
                                            effects = gr.CheckboxGroup(
                                                [
                                                    "Vignette",
                                                    "Scanlines",
                                                    "Noise",
                                                ],
                                                label="Post Effects",
                                            )
                                            custom_elem = gr.Dropdown(
                                                [
                                                    "None",
                                                    "Custom Text",
                                                    "Logo Image",
                                                    "Spectrum Circle",
                                                ],
                                                label="Custom Element",
                                            )
                                    ce_x = gr.Slider(
                                        0, 1, 0.5, label="Element X"
                                    )
                                    ce_y = gr.Slider(
                                        0, 1, 0.5, label="Element Y"
                                    )
                                    ce_scale = gr.Slider(
                                        0.1, 5, 1, label="Element Scale"
                                    )
                                    ce_opacity = gr.Slider(
                                        0, 1, 1, label="Element Opacity"
                                    )
                                    ce_text = gr.Textbox(label="Custom Text")
                                    ce_logo = gr.File(label="Custom Logo")

                                    btn = gr.Button("Generate Video")
                                    out_vid = gr.Video(label="Output Video")
                                    bind_progress_click(
                                        btn,
                                        generate_video_handler,
                                        progress_output=progress_status,
                                        inputs=[
                                            audio_in,
                                            image_in,
                                            style_dd,
                                            resolution,
                                            fps,
                                            sensitivity,
                                            reactivity,
                                            palette,
                                            overlays,
                                            effects,
                                            custom_elem,
                                            ce_x,
                                            ce_y,
                                            ce_scale,
                                            ce_opacity,
                                            ce_text,
                                            ce_logo,
                                        ],
                                        outputs=[out_vid],
                                        action_label="Generate Video",
                                        steps=(
                                            "Validate media",
                                            "Render composition",
                                            "Publish result",
                                        ),
                                        running_detail="Rendering the video composition.",
                                        success_detail="Video render is ready.",
                                    )
                if "lyrics" in selected_tabs:
                    with gr.TabItem("Lyric Video"):
                        with gr.Group():
                            gr.Markdown("### ðŸ“ Lyric Video Creator")
                            lv_audio = gr.Audio(
                                label="Audio File", type="filepath"
                            )
                            lv_bg = gr.Image(
                                label="Background Image", type="filepath"
                            )
                            lv_lyrics = gr.Textbox(label="Lyrics", lines=6)
                            lv_pos = gr.Dropdown(
                                ["top", "center", "bottom"],
                                value="bottom",
                                label="Text Position",
                            )
                            lv_max = gr.Number(value=640, label="Max Dimension")
                            lv_font = gr.Number(value=70, label="Font Size")
                            lv_color = gr.Textbox(
                                value="white", label="Text Color"
                            )
                            lv_stroke = gr.Textbox(
                                value="black", label="Stroke Color"
                            )
                            lv_width = gr.Slider(
                                0, 10, value=2, label="Stroke Width"
                            )
                            lv_fade = gr.Slider(
                                0.0, 5.0, value=0.5, label="Fade Duration"
                            )
                            lv_btn = gr.Button("Make Lyric Video")
                            lv_out = gr.Video(label="Lyric Output")
                            bind_progress_click(
                                lv_btn,
                                lyric_video,
                                progress_output=progress_status,
                                inputs=[
                                    lv_audio,
                                    lv_bg,
                                    lv_lyrics,
                                    lv_pos,
                                    lv_max,
                                    lv_font,
                                    lv_color,
                                    lv_stroke,
                                    lv_width,
                                    lv_fade,
                                ],
                                outputs=[lv_out],
                                action_label="Make Lyric Video",
                                steps=(
                                    "Validate media",
                                    "Render lyric video",
                                    "Publish result",
                                ),
                                running_detail="Rendering the lyric video.",
                                success_detail="Lyric video is ready.",
                            )
                if "visualizer" in selected_tabs:
                    with gr.TabItem("Visualizer"):
                        with gr.Group():
                            gr.Markdown("### ðŸŽ¶ Music Visualizer")
                            mv_audio = gr.Audio(
                                label="Audio File", type="filepath"
                            )
                            mv_width = gr.Number(value=1920, label="Width")
                            mv_height = gr.Number(value=1080, label="Height")
                            mv_fps = gr.Slider(
                                minimum=1, maximum=60, value=30, label="FPS"
                            )
                            mv_btn = gr.Button("Generate Visualizer")
                            mv_out = gr.Video(label="Visualizer Output")
                            bind_progress_click(
                                mv_btn,
                                music_video,
                                progress_output=progress_status,
                                inputs=[mv_audio, mv_width, mv_height, mv_fps],
                                outputs=[mv_out],
                                action_label="Generate Visualizer",
                                steps=(
                                    "Validate audio",
                                    "Render visualizer",
                                    "Publish result",
                                ),
                                running_detail="Rendering the music visualizer.",
                                success_detail="Visualizer video is ready.",
                            )
        launch_blocks(
            app,
            custom_css=shared_css() + "\n" + VideoApp.VIDEO_APP_CSS,
            custom_theme=video_theme,
        )


launch_video_app = VideoApp.launch_video_app
