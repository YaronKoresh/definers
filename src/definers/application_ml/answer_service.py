class AnswerService:
    @staticmethod
    def answer(history, runtime, dependency_loader=None):
        from definers.application_ml.answer_dependency_loader import (
            AnswerDependencyLoader,
        )
        from definers.application_ml.answer_generation_service import (
            AnswerGenerationService,
        )
        from definers.application_ml.answer_history_preparer import (
            AnswerHistoryPreparer,
        )

        processor = runtime.PROCESSORS.get("answer")
        model = runtime.MODELS.get("answer")
        if model is None:
            from definers.ml import init_pretrained_model

            init_pretrained_model("answer")
            processor = runtime.PROCESSORS.get("answer")
            model = runtime.MODELS.get("answer")
        if model is None:
            return None
        if dependency_loader is None:
            dependency_loader = AnswerDependencyLoader()
        prepared_history, image_items, audio_items = (
            AnswerHistoryPreparer.prepare_answer_history(
                history,
                runtime,
                dependency_loader,
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
