import definers.system.runtime_budget as runtime_budget


def test_detect_hosted_runtime_prefers_zerogpu_markers():
    runtime = runtime_budget.detect_hosted_runtime(
        {
            "ZEROGPU": "1",
            "SPACE_ID": "owner/demo",
        }
    )

    assert runtime == "zerogpu"


def test_detect_hosted_runtime_uses_spaces_markers_without_zerogpu():
    runtime = runtime_budget.detect_hosted_runtime(
        {
            "SPACE_HOST": "owner-demo.hf.space",
        }
    )

    assert runtime == "huggingface-spaces"


def test_hosted_runtime_limits_are_stable():
    assert runtime_budget.hosted_preview_row_limit("zerogpu") == 10001
    assert runtime_budget.hosted_guided_row_limit("zerogpu") == 10000
    assert runtime_budget.hosted_guided_media_file_limit("zerogpu") == 64
    assert (
        runtime_budget.hosted_preview_row_limit("huggingface-spaces") == 50001
    )
    assert runtime_budget.hosted_guided_row_limit("huggingface-spaces") == 50000
    assert (
        runtime_budget.hosted_guided_media_file_limit("huggingface-spaces")
        == 256
    )
    assert runtime_budget.hosted_preview_row_limit("local") is None


def test_estimate_session_retention_seconds_respects_runtime_defaults():
    assert (
        runtime_budget.estimate_session_retention_seconds({"ZEROGPU": "1"})
        == 1800.0
    )
    assert (
        runtime_budget.estimate_session_retention_seconds(
            {"SPACE_ID": "owner/demo"}
        )
        == 7200.0
    )
    assert runtime_budget.estimate_session_retention_seconds({}) == 86400.0


def test_explicit_retention_override_beats_runtime_default():
    value = runtime_budget.estimate_session_retention_seconds(
        {
            "ZEROGPU": "1",
            "DEFINERS_GUI_SESSION_RETENTION_SECONDS": "42",
        }
    )

    assert value == 42.0


def test_should_cleanup_after_guided_training_matches_hosted_runtimes():
    assert runtime_budget.should_cleanup_after_guided_training("local") is False
    assert (
        runtime_budget.should_cleanup_after_guided_training("zerogpu") is True
    )
    assert (
        runtime_budget.should_cleanup_after_guided_training(
            "huggingface-spaces"
        )
        is True
    )
