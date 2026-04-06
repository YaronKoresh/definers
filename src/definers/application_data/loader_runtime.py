from definers.logger import init_logger

logger = init_logger()


class LoaderRuntimeSupport:
    @staticmethod
    def _safe_path(path: str):
        """
        Normalize and validate a filesystem path against a safe root.

        Returns the absolute normalized path if it is within the configured
        data root and exists; otherwise returns None.
        """
        import os

        if not path:
            return None
        try:
            base_root = os.environ.get("DEFINERS_DATA_ROOT", os.getcwd())
            base_root = os.path.abspath(base_root)
            full_path = os.path.abspath(path)
            # Ensure the target path is inside the allowed root directory.
            if os.path.commonpath([base_root, full_path]) != base_root:
                logger.error("Path outside allowed root: %s", full_path)
                return None
            if not os.path.exists(full_path):
                logger.error("Path does not exist: %s", full_path)
                return None
            return full_path
        except Exception as error:
            logger.exception("Error validating path %s: %s", path, error)
            return None

    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            import logging

            logging.error(str(error))

    @staticmethod
    def delete(path: str | None) -> None:
        if not path:
            return
        try:
            from definers.system import delete as runtime_delete

            runtime_delete(path)
            return
        except Exception:
            return None

    @staticmethod
    def read(path: str):
        # Validate and normalize path before accessing the filesystem.
        safe_path = LoaderRuntimeSupport._safe_path(path)
        if safe_path is None:
            return None
        try:
            from definers.system import read as runtime_read

            return runtime_read(safe_path)
        except Exception:
            import os

            if os.path.isdir(safe_path):
                return [
                    os.path.join(safe_path, name)
                    for name in os.listdir(safe_path)
                ]
            try:
                with open(safe_path, encoding="utf-8") as file:
                    return file.read()
            except OSError:
                return None

    @staticmethod
    def tmp(extension: str) -> str:
        try:
            from definers.system import tmp as runtime_tmp

            return runtime_tmp(extension)
        except Exception:
            import os
            import tempfile

            suffix = extension if extension.startswith(".") else f".{extension}"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            return path

    @staticmethod
    def runtime():
        from types import SimpleNamespace

        import definers.application_data.arrays as arrays_module
        import definers.application_data.tokenization as tokenization_module

        return SimpleNamespace(
            convert_tensor_dtype=arrays_module.convert_tensor_dtype,
            cupy_to_numpy=arrays_module.cupy_to_numpy,
            get_max_shapes=arrays_module.get_max_shapes,
            init_tokenizer=tokenization_module.init_tokenizer,
            logger=logger,
            reshape_numpy=arrays_module.reshape_numpy,
            tokenize_and_pad=tokenization_module.tokenize_and_pad,
        )
