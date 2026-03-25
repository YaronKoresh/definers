class AnswerHistoryPreparer:
    @classmethod
    def prepare_answer_history(
        cls,
        history,
        runtime,
        image_module,
        soundfile_module,
        librosa_module,
    ):
        from definers.application_ml.answer_audio_loader import AnswerAudioLoader
        from definers.application_ml.answer_content_path_resolver import (
            AnswerContentPathResolver,
        )
        from definers.application_ml.answer_image_loader import AnswerImageLoader
        from definers.application_ml.answer_text_service import AnswerTextService
        from definers.system import get_ext, read

        required_lang = "en"
        image_items = []
        audio_items = []
        prepared_history = [
            {"role": "system", "content": runtime.SYSTEM_MESSAGE}
        ]
        for message in history:
            content = message["content"]
            role = message["role"]
            add_content = ""
            is_text = not isinstance(content, dict) and not isinstance(content, tuple)
            if is_text:
                add_content = AnswerTextService.normalize_answer_text(
                    content,
                    required_lang,
                )
            else:
                for path in AnswerContentPathResolver.content_paths(content):
                    extension = get_ext(path)
                    if extension in runtime.common_audio_formats:
                        loaded_audio = AnswerAudioLoader.read_answer_audio(
                            path,
                            soundfile_module,
                            librosa_module,
                        )
                        if loaded_audio is not None:
                            audio_items.append(loaded_audio)
                            add_content += f" <|audio_{len(audio_items)}|>"
                    elif extension in runtime.iio_formats:
                        image = AnswerImageLoader.read_answer_image(
                            path,
                            image_module,
                        )
                        if image is not None:
                            image_items.append(image)
                            add_content += f" <|image_{len(image_items)}|>"
                    else:
                        file_content = read(path)
                        add_content += "\n\n" + AnswerTextService.normalize_answer_text(
                            file_content,
                            required_lang,
                        )
            AnswerTextService.append_history_message(
                prepared_history,
                role,
                add_content,
            )
        return prepared_history, image_items, audio_items