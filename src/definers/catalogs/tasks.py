from types import MappingProxyType


TASKS = MappingProxyType(
    {
        "general-tokenizer": "distilbert-base-uncased",
        "video": "hunyuanvideo-community/HunyuanVideo-I2V",
        "image": "black-forest-labs/FLUX.1-dev",
        "image-spro": "tencent/SRPO",
        "detect": "facebook/detr-resnet-50",
        "answer": "microsoft/Phi-4-multimodal-instruct",
        "summary": "google-t5/t5-large",
        "music": "facebook/musicgen-small",
        "translate": "facebook/nllb-200-3.3B",
        "song": "https://huggingface.co/tencent/SongGeneration/resolve/main/ckpt/songgeneration_base/model.pt",
        "speech-recognition": "openai/whisper-large-v3",
        "audio-classification": "MIT/ast-finetuned-audioset-10-10-0.4593",
    }
)