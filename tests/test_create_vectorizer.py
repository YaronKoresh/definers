import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

from definers import create_vectorizer


class TestCreateVectorizer(unittest.TestCase):

    def test_create_vectorizer_basic(self):
        texts = ["hello world", "this is a test"]
        vectorizer = create_vectorizer(texts)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertIsNotNone(vectorizer.vocabulary_)
        self.assertIn("hello", vectorizer.vocabulary_)
        self.assertIn("world", vectorizer.vocabulary_)
        self.assertIn("this", vectorizer.vocabulary_)
        self.assertIn("is", vectorizer.vocabulary_)
        self.assertIn("a", vectorizer.vocabulary_)
        self.assertIn("test", vectorizer.vocabulary_)

    def test_create_vectorizer_empty_list(self):
        texts = []
        vectorizer = create_vectorizer(texts)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertEqual(len(vectorizer.vocabulary_), 0)

    def test_create_vectorizer_single_doc(self):
        texts = ["a single document"]
        vectorizer = create_vectorizer(texts)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertEqual(len(vectorizer.vocabulary_), 3)
        self.assertIn("single", vectorizer.vocabulary_)

    def test_create_vectorizer_with_special_chars(self):
        texts = ["email@example.com", "123 numbers", "!@#$%^&*()"]
        vectorizer = create_vectorizer(texts)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertIn("email", vectorizer.vocabulary_)
        self.assertIn("example", vectorizer.vocabulary_)
        self.assertIn("com", vectorizer.vocabulary_)
        self.assertIn("123", vectorizer.vocabulary_)
        self.assertIn("numbers", vectorizer.vocabulary_)
        self.assertEqual(len(vectorizer.vocabulary_), 5)


if __name__ == "__main__":
    unittest.main()
