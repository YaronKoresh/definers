import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "strip_comments.py"
MODULE_SPEC = importlib.util.spec_from_file_location("strip_comments_script", MODULE_PATH)
strip_comments = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(strip_comments)


class TestStripComments(unittest.TestCase):
    def test_remove_docstrings_preserves_empty_suites(self):
        source = textwrap.dedent(
            """
            class EmptyProtocol:
                \"\"\"Marker\"\"\"

            def build_value():
                \"\"\"Factory\"\"\"
            """
        ).strip()

        cleaned = strip_comments.remove_docstrings(source)

        self.assertEqual(
            cleaned,
            textwrap.dedent(
                """
                class EmptyProtocol:
                    pass

                def build_value():
                    pass
                """
            ).strip(),
        )

    def test_remove_docstrings_preserves_tab_indentation(self):
        source = 'class EmptyProtocol:\n\t"""Marker"""\n'

        cleaned = strip_comments.remove_docstrings(source)

        self.assertEqual(cleaned, "class EmptyProtocol:\n\tpass\n")

    def test_remove_docstrings_keeps_protocol_ellipsis_bodies(self):
        source = textwrap.dedent(
            """
            from typing import Protocol

            class ExamplePort(Protocol):
                def run(
                    self,
                    value: str,
                ) -> str: ...
            """
        ).strip()

        cleaned = strip_comments.remove_docstrings(source)

        self.assertEqual(cleaned, source)

    def test_strip_comments_and_format_writes_pass_for_empty_protocol(self):
        source = textwrap.dedent(
            """
            from typing import Protocol

            class EmptyProtocol(Protocol):
                \"\"\"Marker\"\"\"
            """
        ).strip()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / "contracts.py"
            file_path.write_text(source, encoding="utf-8")

            strip_comments.strip_comments_and_format(file_path)

            self.assertEqual(
                file_path.read_text(encoding="utf-8").strip(),
                textwrap.dedent(
                    """
                    from typing import Protocol

                    class EmptyProtocol(Protocol):
                        pass
                    """
                ).strip(),
            )

    def test_strip_comments_and_format_preserves_tab_indentation(self):
        source = 'from typing import Protocol\n\nclass EmptyProtocol(Protocol):\n\t"""Marker"""\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / "contracts.py"
            file_path.write_text(source, encoding="utf-8")

            strip_comments.strip_comments_and_format(file_path)

            self.assertEqual(
                file_path.read_text(encoding="utf-8"),
                "from typing import Protocol\n\nclass EmptyProtocol(Protocol):\n\tpass\n",
            )


if __name__ == "__main__":
    unittest.main()