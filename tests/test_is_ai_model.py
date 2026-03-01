import unittest
from definers import is_ai_model


class TestIsAiModel(unittest.TestCase):
    def test_safetensors(self):
        self.assertTrue(is_ai_model("model.safetensors"))

    def test_onnx(self):
        self.assertTrue(is_ai_model("model.onnx"))

    def test_pt(self):
        self.assertTrue(is_ai_model("model.pt"))

    def test_non_model(self):
        self.assertFalse(is_ai_model("document.txt"))

    def test_python_file(self):
        self.assertFalse(is_ai_model("script.py"))


if __name__ == "__main__":
    unittest.main()
