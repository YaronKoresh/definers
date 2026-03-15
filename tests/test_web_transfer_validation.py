from typing import cast

import pytest

from definers.media.web_transfer import validate_network_url


def test_validate_network_url_accepts_http_and_https() -> None:
    validate_network_url("http://example.com")
    validate_network_url("https://example.com/path?q=1")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (123, "url must be a string"),
        ("ftp://example.com/file.bin", "unsupported URL scheme"),
        ("https://" + "a" * 2000, "url too long"),
    ],
)
def test_validate_network_url_rejects_invalid_input(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_network_url(cast(str, value))
