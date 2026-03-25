class AnswerService:
    @staticmethod
    def answer(history, runtime):
        from PIL import Image
        from definers.application_ml.answer_generation_service import (
            AnswerGenerationService,
        )
        from definers.application_ml.answer_history_preparer import (
            AnswerHistoryPreparer,
        )

        try:
            import librosa
        except Exception:
            librosa = None
        try:
            import soundfile as sf
        except Exception:
            sf = None
        processor = runtime.PROCESSORS.get("answer")
        model = runtime.MODELS.get("answer")
        if model is None:
            return None
        prepared_history, image_items, audio_items = (
            AnswerHistoryPreparer.prepare_answer_history(
                history,
                runtime,
                Image,
                sf,
                librosa,
            )
        )
        if processor is None:
            return AnswerGenerationService.generate_answer_without_processor(
                model,
                prepared_history,
                image_items,
                audio_items,
            )
        return AnswerGenerationService.generate_answer_with_processor(
            processor,
            model,
            prepared_history,
            image_items,
            audio_items,
        )