class AnswerAudioLoader:
    @staticmethod
    def read_answer_audio(path: str, soundfile_module, librosa_module):
        from definers.audio import audio_preview

        loaded_audio = None
        if soundfile_module is not None:
            try:
                loaded_audio = soundfile_module.read(path)
            except Exception:
                loaded_audio = None
        if loaded_audio is None and librosa_module is not None:
            try:
                preview = audio_preview(file_path=path, max_duration=16)
                source = preview if preview else path
                loaded_audio = librosa_module.load(source, sr=16000, mono=True)
            except Exception:
                loaded_audio = None
        return loaded_audio
