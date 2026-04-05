class TokenizationService:
    @staticmethod
    def runtime():
        import definers.application_data.arrays as arrays_module

        return arrays_module

    @staticmethod
    def row_to_text(row) -> str:
        import numpy as np

        features_strings: list[str] = []
        for value in row.values():
            if isinstance(value, (list, np.ndarray)):
                features_strings.extend(map(str, value))
            elif value is not None:
                features_strings.append(str(value))
        return " ".join(features_strings)

    @staticmethod
    def init_tokenizer(
        model_name: str | None = None,
        tokenizer_type: str | None = None,
    ):
        from definers.constants import TOKENIZERS

        model_name = model_name or "google-bert/bert-base-multilingual-cased"
        tokenizer_type = tokenizer_type or "general"
        current_model = TOKENIZERS.get(tokenizer_type, {}).get("model_name")
        if (
            not TOKENIZERS[tokenizer_type]["tokenizer"]
        ) or current_model != model_name:
            from transformers import AutoTokenizer

            TOKENIZERS[tokenizer_type] = {
                "tokenizer": AutoTokenizer.from_pretrained(model_name),
                "model_name": model_name,
            }
        return TOKENIZERS[tokenizer_type]["tokenizer"]

    @classmethod
    def tokenize_and_pad(cls, rows, tokenizer=None):
        runtime = cls.runtime()
        active_tokenizer = tokenizer
        if active_tokenizer is None:
            active_tokenizer = init_tokenizer()
        features_list: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                features_list.append(cls.row_to_text(row))
            elif isinstance(row, str):
                features_list.append(row)
            else:
                return rows
        tokenized_inputs = active_tokenizer(
            features_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return runtime.two_dim_numpy(tokenized_inputs["input_ids"])

    @staticmethod
    def tokenize_or_vectorize(data, tokenizer=None):
        import numpy as np

        if data is None:
            return data
        try:
            array_value = np.array(data)
            if array_value.dtype.kind in ("U", "S", "O"):
                return tokenize_and_pad(array_value.tolist(), tokenizer)
        except Exception:
            return data
        return data


init_tokenizer = TokenizationService.init_tokenizer
tokenize_and_pad = TokenizationService.tokenize_and_pad
tokenize_or_vectorize = TokenizationService.tokenize_or_vectorize
