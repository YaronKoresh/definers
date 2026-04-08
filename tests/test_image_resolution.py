import tempfile
import unittest

from definers.image.helpers import image_resolution


class TestImageResolution(unittest.TestCase):
    def _make_image(self, width, height):
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(128, 64, 32))
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(f.name)
        return f.name

    def test_returns_tuple(self):
        path = self._make_image(100, 200)
        result = image_resolution(path)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_correct_width_and_height(self):
        path = self._make_image(320, 240)
        (w, h) = image_resolution(path)
        self.assertEqual(w, 320)
        self.assertEqual(h, 240)

    def test_square_image(self):
        path = self._make_image(64, 64)
        (w, h) = image_resolution(path)
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)

    def test_tall_image(self):
        path = self._make_image(100, 400)
        (w, h) = image_resolution(path)
        self.assertEqual(w, 100)
        self.assertEqual(h, 400)


if __name__ == "__main__":
    unittest.main()
