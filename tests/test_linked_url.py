import base64
import unittest

from definers import linked_url


class TestLinkedUrl(unittest.TestCase):

    def test_basic_url(self):
        url = "http://example.com"
        data_url = linked_url(url)
        self.assertTrue(
            data_url.startswith(
                "data:text/html;charset=utf-8;base64,"
            )
        )

        encoded_html = data_url.split(",", 1)[1]
        decoded_html = base64.b64decode(encoded_html).decode("utf-8")

        self.assertIn(
            '<base href="http://example.com" target="_top">',
            decoded_html,
        )
        self.assertIn('<a href=""></a>', decoded_html)
        self.assertIn(
            "onload='document.querySelector(\"a\").click()'",
            decoded_html,
        )

    def test_url_with_query_params(self):
        url = "https://example.com/path?param1=value1&param2=value2"
        data_url = linked_url(url)
        self.assertTrue(
            data_url.startswith(
                "data:text/html;charset=utf-8;base64,"
            )
        )

        encoded_html = data_url.split(",", 1)[1]
        decoded_html = base64.b64decode(encoded_html).decode("utf-8")

        self.assertIn(
            '<base href="https://example.com/path" target="_top">',
            decoded_html,
        )
        self.assertIn(
            '<a href="?param1=value1&param2=value2"></a>',
            decoded_html,
        )

    def test_url_without_protocol(self):
        url = "example.com"
        data_url = linked_url(url)
        encoded_html = data_url.split(",", 1)[1]
        decoded_html = base64.b64decode(encoded_html).decode("utf-8")
        self.assertIn(
            '<base href="example.com" target="_top">', decoded_html
        )
        self.assertIn('<a href=""></a>', decoded_html)

    def test_empty_url(self):
        url = ""
        data_url = linked_url(url)
        encoded_html = data_url.split(",", 1)[1]
        decoded_html = base64.b64decode(encoded_html).decode("utf-8")
        self.assertIn('<base href="" target="_top">', decoded_html)
        self.assertIn('<a href=""></a>', decoded_html)


if __name__ == "__main__":
    unittest.main()
