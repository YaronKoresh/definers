from types import ModuleType

from definers.ui.apps import train_ui


def test_train_studio_metadata_covers_ml_surface():
    titles = {section["title"] for section in train_ui.train_studio_sections()}

    assert titles == {
        "Diagnostics And Routing",
        "Model Execution",
        "Runtime Bootstrap",
        "Text And Prompt Lab",
        "Training Orchestration",
    }
    markdown = train_ui.build_capability_markdown().lower()
    for keyword in [
        "training plan preview",
        "task-based inference",
        "answer runtime invocation",
        "text feature extraction",
        "map-reduce summary",
        "ml health snapshot",
        "k-means advisor",
        "rvc checkpoint lookup",
        "init_model_file surface",
        "init_pretrained_model surface",
    ]:
        assert keyword in markdown


def test_train_studio_tab_names_are_stable():
    assert train_ui.train_studio_tab_names() == (
        "Studio",
        "Train",
        "Run",
        "Text Lab",
        "Ops",
    )


class _FakeComponent:
    def __init__(self, registry, kind, *args, **kwargs):
        self.registry = registry
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.click_calls = []
        self.change_calls = []
        registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.click_calls.append((fn, inputs, outputs, kwargs))
        return self

    def change(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.change_calls.append((fn, inputs, outputs, kwargs))
        return self


class _FakeGoogleFont:
    def __init__(self, name):
        self.name = name


class _FakeTheme:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.settings = None

    def set(self, **kwargs):
        self.settings = kwargs
        return self


def _build_fake_gradio_module(registry):
    fake_gradio = ModuleType("gradio")
    fake_gradio.themes = ModuleType("gradio.themes")
    fake_gradio.themes.GoogleFont = _FakeGoogleFont
    fake_gradio.themes.Soft = _FakeTheme
    fake_gradio.themes.colors = ModuleType("gradio.themes.colors")
    fake_gradio.themes.colors.emerald = "emerald"
    fake_gradio.themes.colors.orange = "orange"
    fake_gradio.themes.colors.stone = "stone"
    component_names = [
        "Accordion",
        "Blocks",
        "Button",
        "Checkbox",
        "Code",
        "Column",
        "Dropdown",
        "File",
        "HTML",
        "Markdown",
        "Number",
        "Radio",
        "Row",
        "Slider",
        "TabItem",
        "Tabs",
        "Textbox",
    ]
    for name in component_names:
        setattr(
            fake_gradio,
            name,
            lambda *args, _name=name, **kwargs: _FakeComponent(
                registry,
                _name,
                *args,
                **kwargs,
            ),
        )
    return fake_gradio


def test_build_train_app_constructs_expected_tabs(monkeypatch):
    registry = []
    fake_gradio = _build_fake_gradio_module(registry)

    monkeypatch.setitem(__import__("sys").modules, "gradio", fake_gradio)
    monkeypatch.setattr(
        "definers.system.install_ffmpeg",
        lambda: None,
    )

    app = train_ui.build_train_app()

    assert app.kind == "Blocks"
    tab_labels = [
        component.kwargs.get("label")
        or (component.args[0] if component.args else None)
        for component in registry
        if component.kind == "TabItem"
    ]
    assert "Studio" in tab_labels
    assert "Train" in tab_labels
    assert "Run" in tab_labels
    assert "Text Lab" in tab_labels
    assert "Ops" in tab_labels
    assert "Guided Mode" in tab_labels
    assert "Advanced Mode" in tab_labels

    open_outputs_button = next(
        component
        for component in registry
        if component.kind == "Button"
        and (component.args[0] if component.args else None)
        == "Open Outputs Folder"
    )
    assert open_outputs_button.click_calls

    assert any(
        component.kind == "Button"
        and component.args
        and component.args[0] == "Inspect My Inputs"
        for component in registry
    )
