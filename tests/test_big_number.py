import unittest

from definers import big_number


class TestBigNumber(unittest.TestCase):

    def test_default_zeros(self):
        self.assertEqual(big_number(), 10000000000)

    def test_zero_zeros(self):
        self.assertEqual(big_number(zeros=0), 1)

    def test_one_zero(self):
        self.assertEqual(big_number(zeros=1), 10)

    def test_five_zeros(self):
        self.assertEqual(big_number(zeros=5), 100000)

    def test_large_number_of_zeros(self):
        self.assertEqual(big_number(zeros=18), 1000000000000000000)

    def test_return_type(self):
        self.assertIsInstance(big_number(), int)


if __name__ == "__main__":
    unittest.main()
