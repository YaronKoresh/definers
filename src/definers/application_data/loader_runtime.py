class LoaderRuntimeSupport:
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
        try:
            from definers.system import read as runtime_read

            return runtime_read(path)
        except Exception:
            import os

            if os.path.isdir(path):
                return [os.path.join(path, name) for name in os.listdir(path)]
            try:
                with open(path, encoding="utf-8") as file:
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
        import definers.data as data_module

        return data_module