import ast
import io
import pathlib
import re
import tokenize


def remove_docstrings(source):
    try:
        tree = ast.parse(source)
        docstring_positions = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Str, ast.Constant))
                ):
                    stmt = node.body[0]
                    docstring_positions.add((stmt.lineno, stmt.end_lineno))

        lines = source.split("\n")
        result_lines = []
        for i, line in enumerate(lines, 1):
            skip = False
            for start, end in docstring_positions:
                if start <= i <= end:
                    skip = True
                    break
            if not skip:
                result_lines.append(line)

        return "\n".join(result_lines)
    except:
        return source


def strip_comments_and_format(file_path):
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

    except Exception as e:
        print(f"Failed {file_path}: {e}")


for p in pathlib.Path(".").rglob("*.py"):
    strip_comments_and_format(p)
