from definers.logger import init_logger

logger = init_logger()


class LoaderRuntimeSupport:
    @staticmethod
    def _resolved_path(path: str) -> str:
        import os
        from pathlib import Path

        return os.path.normcase(str(Path(path).expanduser().resolve()))

    @classmethod
    def _trusted_roots(cls) -> list[str]:
        import os
        import tempfile

        roots = [os.getcwd(), tempfile.gettempdir()]
        configured_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
        if configured_root:
            roots.insert(0, configured_root)
        trusted_roots = []
        for root in roots:
            try:
                trusted_roots.append(cls._resolved_path(root))
            except Exception:
                continue
        return trusted_roots

    @classmethod
    def _is_trusted_path(cls, path: str) -> bool:
        import os

        resolved_path = cls._resolved_path(path)
        for root in cls._trusted_roots():
            try:
                if os.path.commonpath([root, resolved_path]) == root:
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _safe_path(path: str):
        import os

        if not path:
            return None
        try:
            full_path = os.path.abspath(path)

            if not LoaderRuntimeSupport._is_trusted_path(full_path):
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
