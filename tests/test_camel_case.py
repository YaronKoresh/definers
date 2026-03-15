import pytest

from definers.text import camel_case


def test_camel_case_basic():
    assert camel_case("hello world") == "helloWorld"


def test_camel_case_symbols():
    assert camel_case("hello-world_test") == "helloWorldTest"


def test_camel_case_empty():
    assert camel_case("") == ""
    assert camel_case(None) == ""


def test_camel_case_single():
    assert camel_case("word") == "word"
