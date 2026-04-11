from definers.ui.apps.translate import TranslateApp


def test_translate_app_resolves_language_code_from_name():
    assert (
        TranslateApp.target_language_code(
            "English",
            {"en": "english", "he": "hebrew"},
        )
        == "en"
    )


def test_translate_app_returns_normalized_input_for_unknown_language():
    assert (
        TranslateApp.target_language_code(
            "  xx-custom  ",
            {"en": "english", "he": "hebrew"},
        )
        == "xx-custom"
    )
