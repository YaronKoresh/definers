from __future__ import annotations

import threading


def big_number(zeros=10):
    return int("1" + "0" * zeros)


def thread(func, *args, **kwargs):
    from definers import system as system_module

    started = threading.Event()

    def wrapper(*inner_args, **inner_kwargs):
        started.set()
        try:
            func(*inner_args, **inner_kwargs)
        except Exception as error:
            system_module.catch(error)

    worker = threading.Thread(
        target=wrapper,
        args=args,
        kwargs=kwargs,
        daemon=True,
    )
    worker.start()
    started.wait(timeout=1)
    return worker


def wait(*threads):
    for thread_obj in threads:
        thread_obj.join()
