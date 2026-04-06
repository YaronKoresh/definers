class DatasetValueLoader:
    @staticmethod
    def load_audio_values(path: str, training: bool):
        import definers
        import definers.application_data.loaders as loaders_module

        transformer = definers.sox.Transformer()
        transformer.rate(32000)
        if training:
            temp_wav_path = loaders_module._tmp("wav")
            transformer.build_file(path, temp_wav_path)
            temp_mp3_path = loaders_module._tmp("mp3")
            definers.remove_silence(temp_wav_path, temp_mp3_path)
            directory_path, _ = definers.split_mp3(temp_mp3_path, 5)
            files = loaders_module._read(directory_path) or [temp_mp3_path]
            values = [
                definers.numpy_to_cupy(
                    definers.extract_audio_features(file_path)
                )
                for file_path in files
            ]
            loaders_module._delete(temp_wav_path)
            loaders_module._delete(temp_mp3_path)
            loaders_module._delete(directory_path)
            return values
        temp_mp3_path = loaders_module._tmp("mp3")
        transformer.build_file(path, temp_mp3_path)
        values = definers.numpy_to_cupy(
            definers.extract_audio_features(temp_mp3_path)
        )
        loaders_module._delete(temp_mp3_path)
        return values

    @staticmethod
    def load_table_values(path: str, extension: str):
        import pandas

        if extension == "csv":
            dataframe = pandas.read_csv(path)
            if dataframe.empty:
                dataframe = pandas.read_csv(path, header=None)
        elif extension == "xlsx":
            dataframe = pandas.read_excel(path)
        else:
            dataframe = pandas.read_json(path)
        return dataframe.values

    @staticmethod
    def load_text_values(path: str):
        import definers
        import definers.application_data.loaders as loaders_module

        text = loaders_module._read(path)
        return definers.numpy_to_cupy(definers.extract_text_features(text))

    @staticmethod
    def load_image_values(path: str):
        import definers

        resized = definers.resize_image(path, 1024, 1024)
        resized_path = resized[0] if isinstance(resized, tuple) else resized
        return definers.numpy_to_cupy(
            definers.extract_image_features(resized_path)
        )

    @staticmethod
    def load_video_values(path: str):
        import definers

        resized_video_file = definers.resize_video(path, 1024, 1024)
        adjusted_fps_file = definers.convert_video_fps(resized_video_file, 24)
        return definers.numpy_to_cupy(
            definers.extract_video_features(adjusted_fps_file)
        )

    @classmethod
    def load_as_numpy(cls, path: str, training: bool = False):
        try:
            import logging

            import definers.application_data.loader_runtime as loader_runtime_module
            from definers.constants import iio_formats

            safe_path = loader_runtime_module.LoaderRuntimeSupport._safe_path(
                path
            )
            if safe_path is None:
                logging.error("Rejected unsafe or invalid path: %s", path)
                return None
            extension = cls.path_extension(safe_path)
            if extension is None:
                logging.error("Invalid path format: %s", safe_path)
                return None
            if extension in ["wav", "mp3"]:
                return cls.load_audio_values(safe_path, training)
            if extension in ["csv", "xlsx", "json"]:
                return cls.load_table_values(safe_path, extension)
            if extension == "txt":
                return cls.load_text_values(safe_path)
            if extension in iio_formats:
                return cls.load_image_values(safe_path)
            return cls.load_video_values(safe_path)
        except Exception as error:
            import definers.application_data.loaders as loaders_module

            loaders_module._catch(error)
            return None

    @staticmethod
    def path_extension(path: str) -> str | None:
        import definers.application_data.loaders as loaders_module

        return loaders_module._path_extension(path)
