import unittest
from unittest.mock import MagicMock, patch

import definers.os_utils as os_utils
import definers.path_utils as path_utils
from definers.media.web_transfer import extract_text

if not hasattr(os_utils, "get_python_version"):
    os_utils.get_python_version = lambda: "3.10"
if not hasattr(os_utils, "get_linux_distribution"):
    os_utils.get_linux_distribution = lambda: "linux"

for _name, _value in {
    "normalize_path": lambda path: str(path),
    "full_path": lambda *parts: "/".join(
        str(part) for part in parts if str(part)
    ),
    "paths": lambda *patterns: [],
    "unique": lambda items: list(dict.fromkeys(items)),
    "cwd": lambda: ".",
    "parent_directory": lambda path: "",
    "path_end": lambda path: str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
    "path_ext": lambda path: (
        "" if "." not in str(path) else "." + str(path).rsplit(".", 1)[-1]
    ),
    "path_name": lambda path: (
        str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    ),
    "tmp": lambda *args, **kwargs: "/tmp/mock",
    "secure_path": lambda path, *args, **kwargs: path,
}.items():
    if not hasattr(path_utils, _name):
        setattr(path_utils, _name, _value)


class TestExtractText(unittest.TestCase):
    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_extract_text_successfully(self, mock_sync_playwright, mock_expect):
        mock_css = MagicMock()
        mock_css.path = "//div[@class='content']"
        mock_element = MagicMock()
        mock_element.text_content.return_value = "Expected Text"
        mock_html = MagicMock()
        mock_html.xpath.return_value = [mock_element]

        mock_page = MagicMock()
        mock_page.content.return_value = (
            '<html><body><div class="content">Expected Text</div></body></html>'
        )
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = mock_browser
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

        with patch.dict(
            extract_text.__globals__,
            {
                "CSSSelector": MagicMock(return_value=mock_css),
                "fromstring": MagicMock(return_value=mock_html),
            },
        ):
            result = extract_text("http://example.com", ".content")
        self.assertEqual(result, "Expected Text")

    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_selector_not_found(self, mock_sync_playwright, mock_expect):
        mock_css = MagicMock()
        mock_css.path = "//div[@class='nonexistent']"
        mock_html = MagicMock()
        mock_html.xpath.return_value = []

        mock_page = MagicMock()
        mock_page.content.return_value = (
            '<html><body><div class="other">Some Text</div></body></html>'
        )
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = mock_browser
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

        with patch.dict(
            extract_text.__globals__,
            {
                "CSSSelector": MagicMock(return_value=mock_css),
                "fromstring": MagicMock(return_value=mock_html),
            },
        ):
            result = extract_text("http://example.com", ".nonexistent")
        self.assertEqual(result, "")

    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_empty_page_content(self, mock_sync_playwright, mock_expect):
        mock_css = MagicMock()
        mock_css.path = "dummy"

        mock_page = MagicMock()
        mock_page.content.return_value = ""
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = mock_browser
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

        with patch.dict(
            extract_text.__globals__,
            {
                "CSSSelector": MagicMock(return_value=mock_css),
                "fromstring": MagicMock(),
            },
        ):
            result = extract_text("http://example.com", ".content")
        self.assertEqual(result, "")

    @patch(
        "playwright.sync_api.sync_playwright",
        side_effect=Exception("Playwright Error"),
    )
    def test_playwright_error(self, mock_sync_playwright):
        with self.assertRaises(Exception):
            extract_text("http://example.com", ".content")


if __name__ == "__main__":
    unittest.main()
