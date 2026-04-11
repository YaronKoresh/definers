from definers.logger import init_logger

logger = init_logger()


class LoaderRuntimeSupport:
    @staticmethod
    def _trusted_roots() -> list[str]:
        import os
        import tempfile

        roots = [os.getcwd(), tempfile.gettempdir()]
        configured_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
        if configured_root:
            roots.insert(0, configured_root)
        trusted_roots = []
        for root in roots:
            normalized_root = str(root).strip()
            if not normalized_root:
                continue
            try:
                trusted_roots.append(
                    os.path.abspath(os.path.expanduser(normalized_root))
                )
            except Exception:
                continue
        return trusted_roots

    @staticmethod
    def _safe_path(path: str):
        import os

        from definers.system import secure_path

        if not path:
            return None
        try:
            safe_path = secure_path(
                str(path).strip(),
                trust=LoaderRuntimeSupport._trusted_roots(),
            )
            if not os.path.exists(safe_path):
                logger.error("Path does not exist: %s", safe_path)
                return None
            return safe_path
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

        import definers.data.arrays as arrays_module
        import definers.data.tokenization as tokenization_module

        return SimpleNamespace(
            convert_tensor_dtype=arrays_module.convert_tensor_dtype,
            cupy_to_numpy=arrays_module.cupy_to_numpy,
            get_max_shapes=arrays_module.get_max_shapes,
            init_tokenizer=tokenization_module.init_tokenizer,
            logger=logger,
            reshape_numpy=arrays_module.reshape_numpy,
            tokenize_and_pad=tokenization_module.tokenize_and_pad,
        )
