class AnswerGenerationService:
    @staticmethod
    def generate_answer_without_processor(
        model,
        history,
        image_items,
        audio_items,
    ):
        prompt = (
            "".join(
                [
                    f"<|{message['role']}|>{message['content']}<|end|>"
                    for message in history
                ]
            )
            + "<|assistant|>"
        )
        generate_kwargs = {
            "prompt": prompt,
            "max_length": 200,
            "beam_width": 16,
        }
        if image_items:
            generate_kwargs["images"] = image_items
        if audio_items:
            generate_kwargs["audios"] = audio_items
        return model.generate(**generate_kwargs)

    @staticmethod
    def generate_answer_with_processor(
        processor,
        model,
        history,
        image_items,
        audio_items,
    ):
        from definers.constants import beam_kwargs
        from definers.cuda import device

        prompt = processor.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt,
            images=image_items if image_items else None,
            audios=audio_items if audio_items else None,
            return_tensors="pt",
        )
        inputs = inputs.to(device())
        generate_ids = model.generate(
            **inputs,
            **beam_kwargs,
            max_length=4096,
            num_logits_to_keep=1,
        )
        output_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
