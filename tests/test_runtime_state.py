from concurrent.futures import ThreadPoolExecutor

from definers.state import (
    create_runtime_state,
    delete_runtime_state,
    get_config,
    get_model,
    get_processor,
    get_runtime_state,
    get_tokenizer,
    get_tokenizer_entry,
    list_runtime_scopes,
    reset_runtime_state,
    set_config,
    set_model,
    set_processor,
    set_tokenizer,
)


def test_runtime_state_model_accessors() -> None:
    reset_runtime_state()
    assert get_model("answer") is None
    marker = object()
    assert set_model("answer", marker) is marker
    assert get_model("answer") is marker


def test_runtime_state_tokenizer_accessors() -> None:
    reset_runtime_state()
    marker = object()
    entry = set_tokenizer("summary", marker, model_name="summary-model")
    assert entry["tokenizer"] is marker
    assert entry["model_name"] == "summary-model"
    assert get_tokenizer("summary") is marker
    assert get_tokenizer_entry("summary") == entry


def test_runtime_state_processor_and_config_accessors() -> None:
    reset_runtime_state()
    processor = object()
    config = {"temperature": 0.7}
    assert set_processor("music", processor) is processor
    assert get_processor("music") is processor
    assert set_config("answer", config) is config
    assert get_config("answer") == config


def test_runtime_state_reset_restores_defaults() -> None:
    state = get_runtime_state()
    state.set_model("summary", object())
    state.set_tokenizer("general", object(), model_name="general-model")
    state.set_processor("music", object())
    state.set_config("answer", {"mode": "custom"})

    reset_runtime_state()

    reset_state = get_runtime_state()
    assert reset_state.get_model("summary") is None
    assert reset_state.get_tokenizer("general") is None
    assert reset_state.get_tokenizer_entry("general") == {
        "tokenizer": None,
        "model_name": None,
    }
    assert reset_state.get_processor("music") is None
    assert reset_state.get_config("answer") is None


def test_named_runtime_scopes_are_isolated() -> None:
    reset_runtime_state()
    delete_runtime_state("chat-session")

    default_state = get_runtime_state()
    scoped_state = create_runtime_state("chat-session", replace=True)

    default_marker = object()
    scoped_marker = object()
    default_state.set_model("answer", default_marker)
    scoped_state.set_model("answer", scoped_marker)

    assert get_runtime_state().get_model("answer") is default_marker
    assert (
        get_runtime_state("chat-session").get_model("answer") is scoped_marker
    )
    assert list_runtime_scopes() == ("chat-session", "default")


def test_reset_runtime_state_can_target_named_scope_only() -> None:
    reset_runtime_state()
    delete_runtime_state("training-job")

    default_state = get_runtime_state()
    scoped_state = create_runtime_state("training-job", replace=True)
    default_marker = object()
    scoped_marker = object()
    default_state.set_processor("music", default_marker)
    scoped_state.set_processor("music", scoped_marker)

    reset_runtime_state("training-job")

    assert get_runtime_state().get_processor("music") is default_marker
    assert get_runtime_state("training-job").get_processor("music") is None


def test_delete_runtime_state_rejects_default_scope() -> None:
    try:
        delete_runtime_state("default")
    except ValueError as error:
        assert str(error) == "default runtime scope cannot be deleted"
    else:
        raise AssertionError("default scope deletion should fail")


def test_replace_default_runtime_state_preserves_default_accessors() -> None:
    reset_runtime_state()
    set_model("answer", "configured")

    replacement = create_runtime_state("default", replace=True)

    assert replacement is get_runtime_state()
    assert get_model("answer") is None


def test_create_runtime_state_rejects_blank_scope() -> None:
    try:
        create_runtime_state("   ", replace=True)
    except ValueError as error:
        assert str(error) == "scope must not be empty"
    else:
        raise AssertionError("blank scope creation should fail")


def test_create_runtime_state_is_stable_under_concurrent_access() -> None:
    delete_runtime_state("parallel-scope")

    with ThreadPoolExecutor(max_workers=8) as executor:
        states = list(
            executor.map(
                lambda _: create_runtime_state("parallel-scope"),
                range(32),
            )
        )

    first_state = states[0]
    assert all(state is first_state for state in states)


def test_runtime_state_supports_concurrent_helper_mutations() -> None:
    scoped_state = create_runtime_state("parallel-mutations", replace=True)

    def mutate(index: int) -> tuple[int, object]:
        marker = object()
        scoped_state.set_model(f"answer-{index}", marker)
        scoped_state.set_tokenizer(
            "summary",
            marker,
            model_name=f"summary-{index}",
        )
        return index, scoped_state.get_model(f"answer-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(mutate, range(32)))

    assert len(results) == 32
    assert all(marker is not None for _, marker in results)
    tokenizer_entry = scoped_state.get_tokenizer_entry("summary")
    assert tokenizer_entry is not None
    assert tokenizer_entry["tokenizer"] is not None
    assert str(tokenizer_entry["model_name"]).startswith("summary-")
