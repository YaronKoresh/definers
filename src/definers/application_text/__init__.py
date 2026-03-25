class ApplicationTextFacade:
    @staticmethod
    def get_language_module():
        import importlib

        return importlib.import_module("definers.application_text.language")

    @staticmethod
    def get_validation_module():
        import importlib

        return importlib.import_module("definers.application_text.validation")

    @classmethod
    def get_language_export(cls, name):
        return getattr(cls.get_language_module(), name)

    @classmethod
    def get_validation_export(cls, name):
        return getattr(cls.get_validation_module(), name)


ai_translate = ApplicationTextFacade.get_language_export("ai_translate")
camel_case = ApplicationTextFacade.get_language_export("camel_case")
duck_translate = ApplicationTextFacade.get_language_export("duck_translate")
google_translate = ApplicationTextFacade.get_language_export("google_translate")
language = ApplicationTextFacade.get_language_export("language")
set_system_message = ApplicationTextFacade.get_language_export(
    "set_system_message"
)
simple_text = ApplicationTextFacade.get_language_export("simple_text")
strip_nikud = ApplicationTextFacade.get_language_export("strip_nikud")
translate_with_code = ApplicationTextFacade.get_language_export(
    "translate_with_code"
)
TextInputValidator = ApplicationTextFacade.get_validation_export(
    "TextInputValidator"
)

__all__ = [
    "ai_translate",
    "camel_case",
    "duck_translate",
    "google_translate",
    "language",
    "set_system_message",
    "simple_text",
    "strip_nikud",
    "TextInputValidator",
    "translate_with_code",
]
