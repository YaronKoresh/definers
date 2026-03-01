from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

INCLUDE_DIRECTORIES = {
    "src",
    "tests",
    "scripts",
}

EXCLUDE_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
}


class DocstringCleaner(ast.NodeTransformer):
    def _strip_first_docstring(self, node: ast.AST) -> ast.AST:
        if not hasattr(node, "body"):
            return node
        body = getattr(node, "body")
        if not isinstance(body, list) or not body:
            return node
        first_node = body[0]
        if not isinstance(first_node, ast.Expr):
            return node
        expression_value = first_node.value
        if isinstance(expression_value, ast.Constant) and isinstance(
            expression_value.value, str
        ):
            body.pop(0)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        return self._strip_first_docstring(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_first_docstring(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_first_docstring(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_first_docstring(node)


def strip_comments(source_text: str) -> str:
    generated_tokens = tokenize.generate_tokens(
        io.StringIO(source_text).readline
    )
    retained_tokens: list[tuple[int, str]] = []
    for token_type, token_text, _, _, _ in generated_tokens:
        if token_type == tokenize.COMMENT:
            continue
        retained_tokens.append((token_type, token_text))
    return tokenize.untokenize(retained_tokens)


def sanitize_python_file(file_path: Path) -> None:
    original_text = file_path.read_text(encoding="utf-8")
    text_without_comments = strip_comments(original_text)
    syntax_tree = ast.parse(text_without_comments)
    cleaned_tree = DocstringCleaner().visit(syntax_tree)
    ast.fix_missing_locations(cleaned_tree)
    rewritten_text = ast.unparse(cleaned_tree)
    file_path.write_text(f"{rewritten_text}\n", encoding="utf-8")


def should_process(file_path: Path) -> bool:
    parts = set(file_path.parts)
    if not INCLUDE_DIRECTORIES.intersection(parts):
        return False
    if EXCLUDE_DIRECTORIES.intersection(parts):
        return False
    return file_path.suffix == ".py"


def run() -> None:
    root = Path(".")
    for file_path in root.rglob("*.py"):
        if should_process(file_path):
            sanitize_python_file(file_path)


if __name__ == "__main__":
    run()
