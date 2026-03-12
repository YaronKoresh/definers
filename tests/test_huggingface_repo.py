import unittest

from definers import is_huggingface_repo


class TestHuggingFaceRepo(unittest.TestCase):
    def test_valid_repo_ids(self):
        self.assertTrue(is_huggingface_repo("user/model"))
        self.assertTrue(is_huggingface_repo("User-123/model_4.5"))
        self.assertTrue(is_huggingface_repo("a/b"))

    def test_invalid_repo_ids(self):
        self.assertFalse(is_huggingface_repo(""))
        self.assertFalse(is_huggingface_repo("noslash"))
        self.assertFalse(is_huggingface_repo("/model"))
        self.assertFalse(is_huggingface_repo("user/"))

        self.assertFalse(is_huggingface_repo("user/model/extra"))

        self.assertFalse(is_huggingface_repo("user$/model"))
        self.assertFalse(is_huggingface_repo("user/model?"))

    def test_non_string_input(self):
        self.assertFalse(is_huggingface_repo(None))
        self.assertFalse(is_huggingface_repo(123))


if __name__ == "__main__":
    unittest.main()
