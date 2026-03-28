class AnswerDependencyLoader:
    @staticmethod
    def load_image_module():
        try:
            from PIL import Image
        except Exception:
            return None
        return Image

    @staticmethod
    def load_soundfile_module():
        try:
            import soundfile as soundfile_module
        except Exception:
            return None
        return soundfile_module

    @staticmethod
    def load_librosa_module():
        try:
            import librosa as librosa_module
        except Exception:
            return None
        return librosa_module