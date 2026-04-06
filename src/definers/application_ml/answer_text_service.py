class AnswerTextService:
    @classmethod
    def normalize_answer_text(cls, text: str, required_lang: str) -> str:
        if required_lang == "en" and text.isascii():
            return text
        from definers.text import ai_translate, language

        try:
            content_lang = language(text)
        except Exception:
            return text
        if content_lang != required_lang:
            try:
                return ai_translate(text, lang=required_lang)
            except Exception:
                return text
        return text

    @staticmethod
    def append_history_message(history_items, role: str, content: str) -> None:
        stripped_content = content.strip()
        if history_items[-1]["role"] != role:
            history_items.append({"role": role, "content": stripped_content})
            return
        history_items[-1]["content"] += "\n\n" + stripped_content
        history_items[-1]["content"] = history_items[-1]["content"].strip()
