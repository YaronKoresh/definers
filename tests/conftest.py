import pytest

def pytest_configure(config):
    try:
        import PIL
        if hasattr(PIL, 'Image') and hasattr(PIL.Image, '__spec__'):
            if not hasattr(PIL, '__spec__'):
                PIL.__spec__ = PIL.Image.__spec__
    except ImportError:
        pass