from .platform.paths import tmp
from definers.platform.filesystem import save
from definers.shared_kernel.observability import catch_exception, log_message


def log(subject: str, data: str | int | None = None, status: bool | str | None = None):
    from definers.logger import logger

    log_message(logger, subject, data, status)


def catch(error: object, message: str | None = None, reraise: bool = False) -> None:
    from definers.logger import logger

    catch_exception(logger, error, message=message, reraise=reraise)


def save_temp_text(text_content):
    if text_content is None:
        return None
    temp_path = tmp("data")
    save(temp_path, str(text_content))
    return temp_path
