import unittest
from unittest.mock import MagicMock, patch

import playwright

import definers
from definers import extract_text


class TestExtractText(unittest.TestCase):

    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_extract_text_successfully(self, mock_sync_playwright):
        mock_page = MagicMock()
        mock_page.content.return_value = '<html><body><div class="content">Expected Text</div></body></html>'
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = (
            mock_browser
        )
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

        result = extract_text("http://example.com", ".content")
        self.assertEqual(result, "Expected Text")

    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_selector_not_found(self, mock_sync_playwright):
        mock_page = MagicMock()
        mock_page.content.return_value = '<html><body><div class="other">Some Text</div></body></html>'
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = (
            mock_browser
        )
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

        result = extract_text("http://example.com", ".nonexistent")
        self.assertEqual(result, "")

    @patch("playwright.sync_api.expect")
    @patch("playwright.sync_api.sync_playwright")
    def test_empty_page_content(self, mock_sync_playwright, mock_expect):
        mock_page = MagicMock()
        mock_page.content.return_value = ""
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.firefox.launch.return_value = (
            mock_browser
        )
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_instance
        )

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
