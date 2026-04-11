import builtins
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

    def load(self, fn=None, inputs=None, outputs=None, **kwargs):
        return self._record_event(
            "load",
            fn,
            inputs,
            outputs,
            **kwargs,
        )


def _build_fake_gradio_module(registry):
    context_stack = []
    fake_gradio = ModuleType("gradio")
    fake_gradio.Error = RuntimeError
    fake_gradio.update = lambda **kwargs: kwargs
    component_names = [
        "Accordion",
        "Audio",
        "Blocks",
        "Button",
        "Checkbox",
        "Column",
        "Dropdown",
        "File",
        "Group",
        "HTML",
        "Image",
        "Markdown",
        "Number",
        "Radio",
        "Row",
        "Slider",
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


def test_launch_audio_app_hides_mastering_output_column_until_result(
    monkeypatch,
):
    registry = []
    fake_gradio = _build_fake_gradio_module(registry)
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    monkeypatch.delitem(sys.modules, "definers.ui.apps.audio", raising=False)
    monkeypatch.delitem(sys.modules, "definers.audio", raising=False)
    monkeypatch.delitem(sys.modules, "definers.audio.feedback", raising=False)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name).split(".", 1)[0] == "librosa":
            raise ModuleNotFoundError(
                "No module named 'librosa'",
                name="librosa",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    audio_app = importlib.import_module("definers.ui.apps.audio")

    monkeypatch.setattr(audio_app, "init_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(audio_app, "launch_blocks", lambda app: None)
    monkeypatch.setattr(
        "definers.ui.apps.audio_workspace.prepare_audio_workspace",
        lambda: {"svc_installed": False},
    )
    monkeypatch.setattr("definers.cuda.device", lambda: "cuda")

    audio_app.launch_audio_app(tool_names=["Mastering Studio"])

    output_column = next(
        component
        for component in registry
        if component.kind == "Column"
        and component.kwargs.get("elem_id") == "enhancer-output-column"
    )
    output_group = next(
        component
        for component in registry
        if component.kind == "Group"
        and component.kwargs.get("elem_id") == "enhancer-output-box"
    )

    assert output_column.kwargs.get("visible") is False
    assert output_group.parent is output_column

    master_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Master Audio"
    )
    slider_labels = {
        component.kwargs.get("label")
        for component in registry
        if component.kind == "Slider"
    }
    assert {
        "Vocal/Other Glue Reverb",
        "Drum Edge / Expand-Compress",
        "Extra Vocal Pullback (dB)",
    } <= slider_labels

    master_input_labels = {
        component.kwargs.get("label")
        for component in master_button.event_calls["click"][0]["args"][1]
        if hasattr(component, "kwargs")
    }
    assert {
        "Vocal/Other Glue Reverb",
        "Drum Edge / Expand-Compress",
        "Extra Vocal Pullback (dB)",
    } <= master_input_labels

    master_outputs = master_button.event_calls["click"][0]["args"][2]
    assert output_column in master_outputs

    labels = {
        _button_label(component)
        for component in registry
        if component.kind == "Button" and component.event_calls.get("click")
    }
    assert {
        "Prepare Staged Job",
        "Run Full Job",
        "Separate Stems",
        "Build Stem Mix",
        "Finalize Master",
        "Refresh Job",
    } <= labels

    clear_buttons = [
        component
        for component in registry
        if component.kind == "Button" and _button_label(component) == "Clear"
    ]
    assert any(
        output_column in click_call["args"][2]
        for button in clear_buttons
        for click_call in button.event_calls.get("click", [])
    )

    open_outputs_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and _button_label(component) == "Open Outputs Folder"
    )
    assert open_outputs_button.event_calls.get("click")
