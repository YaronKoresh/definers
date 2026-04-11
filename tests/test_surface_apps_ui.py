import importlib
import sys
from types import ModuleType


class _FakeComponent:
    def __init__(self, registry, context_stack, kind, *args, **kwargs):
        self.registry = registry
        self.context_stack = context_stack
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.children = []
        self.parent = context_stack[-1] if context_stack else None
        self.event_calls = {}
        if self.parent is not None:
            self.parent.children.append(self)
        registry.append(self)

    def __enter__(self):
        self.context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.context_stack and self.context_stack[-1] is self:
            self.context_stack.pop()
        return False

    def _record_event(self, name, *args, **kwargs):
        self.event_calls.setdefault(name, []).append(
            {"args": args, "kwargs": kwargs}
        )
        return self

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        return self._record_event(
            "click",
            fn,
            inputs,
            outputs,
            **kwargs,
        )

    def change(self, fn=None, inputs=None, outputs=None, **kwargs):
        return self._record_event(
            "change",
            fn,
            inputs,
            outputs,
            **kwargs,
        )


class _FakeTheme:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _build_fake_gradio_module(registry):
    context_stack = []
    fake_gradio = ModuleType("gradio")
    fake_gradio.Error = RuntimeError
    fake_gradio.update = lambda **kwargs: kwargs
    fake_gradio.themes = ModuleType("gradio.themes")
    fake_gradio.themes.Base = _FakeTheme
    component_names = [
        "Accordion",
        "Audio",
        "Blocks",
        "Button",
        "Chatbot",
        "ChatInterface",
        "Checkbox",
        "CheckboxGroup",
        "Column",
        "Dropdown",
        "File",
        "Group",
        "HTML",
        "Image",
        "Markdown",
        "MultimodalTextbox",
        "Number",
        "Radio",
        "Row",
        "Slider",
        "State",
        "TabItem",
        "Tabs",
        "Textbox",
        "Video",
    ]
    for name in component_names:
        setattr(
            fake_gradio,
            name,
            lambda *args, _name=name, **kwargs: _FakeComponent(
                registry,
                context_stack,
                _name,
                *args,
                **kwargs,
            ),
        )
    return fake_gradio


def _button_label(component):
    if component.kwargs.get("value") is not None:
        return component.kwargs["value"]
    if component.args:
        return component.args[0]
    return None


def _patch_shared_launch(monkeypatch):
    shared = importlib.import_module("definers.ui.gradio_shared")
    monkeypatch.setattr(shared, "launch_blocks", lambda *args, **kwargs: None)


def test_launch_image_app_binds_progress_actions(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)

    image_app = importlib.import_module("definers.ui.apps.image")
    image_app.launch_image_app()

    labels = {
        _button_label(component)
        for component in registry
        if component.kind == "Button" and component.event_calls.get("click")
    }
    assert {
        "Generate",
        "Upscale",
        "Add title(s)",
        "Open Outputs Folder",
    } <= labels


def test_launch_image_generate_jobs_app_binds_guided_actions(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)

    image_jobs = importlib.import_module("definers.ui.apps.image_generate_jobs")
    image_jobs.launch_image_generate_jobs_app()

    labels = {
        _button_label(component)
        for component in registry
        if component.kind == "Button" and component.event_calls.get("click")
    }
    assert {
        "1. Prepare Job",
        "2. Generate Image",
        "3. Upscale Result",
        "4. Add Titles",
        "Refresh Job",
        "Open Outputs Folder",
    } <= labels


def test_launch_translate_app_binds_translate_action(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)

    translate_app = importlib.import_module("definers.ui.apps.translate")
    translate_app.launch_translate_app()

    translate_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Translate"
    )
    assert translate_button.event_calls.get("click")
    open_outputs_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Open Outputs Folder"
    )
    assert open_outputs_button.event_calls.get("click")


def test_launch_faiss_app_binds_build_action(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)
    fake_ml = ModuleType("definers.ml")
    fake_ml.build_faiss = lambda: "faiss.whl"
    monkeypatch.setitem(sys.modules, "definers.ml", fake_ml)

    faiss_app = importlib.import_module("definers.ui.apps.faiss")
    faiss_app.launch_faiss_app()

    build_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Build FAISS Wheel"
    )
    assert build_button.event_calls.get("click")
    open_outputs_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Open Outputs Folder"
    )
    assert open_outputs_button.event_calls.get("click")


def test_launch_chat_app_exposes_baseline_shell(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)
    fake_chat_handlers = ModuleType("definers.ui.chat_handlers")
    fake_chat_handlers.get_chat_response_stream = lambda *args, **kwargs: "ok"
    monkeypatch.setitem(
        sys.modules,
        "definers.ui.chat_handlers",
        fake_chat_handlers,
    )

    chat_app = importlib.import_module("definers.ui.apps.chat_app")
    chat_app.launch_chat_app()

    open_outputs_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Open Outputs Folder"
    )
    assert open_outputs_button.event_calls.get("click")
    assert any(
        component.kind == "Markdown"
        and "Assistant ready" in str(component.kwargs.get("value", ""))
        for component in registry
    )


def test_launch_animation_app_binds_chunk_actions(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)
    monkeypatch.setattr("definers.system.tmp", lambda dir=False: "chunks")

    animation_app = importlib.import_module("definers.ui.apps.animation")
    animation_app.launch_animation_app()

    labels = {
        _button_label(component)
        for component in registry
        if component.kind == "Button" and component.event_calls.get("click")
    }
    assert {
        "Generate Next Chunk",
        "Combine Chunks into Final GIF",
        "Start Over",
        "Open Outputs Folder",
    } <= labels


def test_launch_video_app_binds_render_actions(monkeypatch):
    registry = []
    monkeypatch.setitem(
        sys.modules, "gradio", _build_fake_gradio_module(registry)
    )
    _patch_shared_launch(monkeypatch)
    fake_video_gui = ModuleType("definers.video.gui")
    fake_video_gui.filter_styles = lambda *args, **kwargs: []
    fake_video_gui.generate_video_handler = lambda *args, **kwargs: "video.mp4"
    fake_lyric_service = ModuleType("definers.ui.lyric_video_service")
    fake_lyric_service.lyric_video = lambda *args, **kwargs: "lyrics.mp4"
    fake_music_service = ModuleType("definers.ui.music_video_service")
    fake_music_service.music_video = lambda *args, **kwargs: "visualizer.mp4"
    monkeypatch.setitem(sys.modules, "definers.video.gui", fake_video_gui)
    monkeypatch.setitem(
        sys.modules,
        "definers.ui.lyric_video_service",
        fake_lyric_service,
    )
    monkeypatch.setitem(
        sys.modules,
        "definers.ui.music_video_service",
        fake_music_service,
    )

    video_app = importlib.import_module("definers.ui.apps.video")
    video_app.launch_video_app()

    labels = {
        _button_label(component)
        for component in registry
        if component.kind == "Button" and component.event_calls.get("click")
    }
    assert {
        "Generate Video",
        "Make Lyric Video",
        "Generate Visualizer",
        "Open Outputs Folder",
    } <= labels
