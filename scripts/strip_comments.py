import ast
import io
import pathlib
import tokenize


def _is_string_docstring_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    legacy_str_node = getattr(ast, "Str", None)
    return isinstance(legacy_str_node, type) and isinstance(
        node, legacy_str_node
    )


def _is_string_docstring_statement(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    return _is_string_docstring_value(node.value)


def remove_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
        source_lines = source.split("\n")
        docstring_replacements: dict[int, tuple[int, list[str]]] = {}

        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.Module,
                ),
            ):
                if node.body and _is_string_docstring_statement(node.body[0]):
                    stmt = node.body[0]
                    replacement: list[str] = []
                    if len(node.body) == 1 and not isinstance(node, ast.Module):
                        indent = source_lines[stmt.lineno - 1][
                            : len(source_lines[stmt.lineno - 1])
                            - len(source_lines[stmt.lineno - 1].lstrip())
                        ]
                        replacement = [f"{indent}pass"]
                    docstring_replacements[stmt.lineno] = (
                        stmt.end_lineno,
                        replacement,
                    )

        lines = source_lines
        result_lines = []
        line_number = 1
        while line_number <= len(lines):
            replacement_info = docstring_replacements.get(line_number)
            if replacement_info is None:
                result_lines.append(lines[line_number - 1])
                line_number += 1
                continue
            end_line, replacement_lines = replacement_info
            result_lines.extend(replacement_lines)
            line_number = end_line + 1

        return "\n".join(result_lines)
    except SyntaxError:
        return source


def strip_comments_and_format(file_path: pathlib.Path) -> None:
    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            source = f.read()

        source = remove_docstrings(source)

        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        result = []

        for toktype, tokval, start, end, line in tokens:
            if toktype == tokenize.COMMENT:
                continue
            result.append((toktype, tokval, start, end, line))

        cleaned = tokenize.untokenize(result)
        lines = cleaned.split("\n")

        blank_count = 0
        final_lines = []
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)

        cleaned = "\n".join(final_lines)

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(cleaned)

        print(f"Stripped: {file_path}")

    except (IndentationError, OSError, tokenize.TokenError) as error:
        print(f"Failed {file_path}: {error}")


def main() -> None:
    for path in pathlib.Path(".").rglob("*.py"):
        strip_comments_and_format(path)


if __name__ == "__main__":
    main()
