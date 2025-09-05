import unittest
from definers import get_max_resolution

class TestGetMaxResolution(unittest.TestCase):

    def test_basic_calculation_16_9(self):
        width, height = 1920, 1080
        new_width, new_height = get_max_resolution(width, height)
        self.assertIsInstance(new_width, int)
        self.assertIsInstance(new_height, int)
        self.assertLessEqual(new_width * new_height, 0.25 * 1000 * 1000)
        self.assertTrue(new_width % 16 == 0)
        self.assertTrue(new_height % 16 == 0)
        self.assertAlmostEqual(width / height, new_width / new_height, places=2)

    def test_basic_calculation_4_3(self):
        width, height = 800, 600
        new_width, new_height = get_max_resolution(width, height)
        self.assertLessEqual(new_width * new_height, 0.25 * 1000 * 1000)
        self.assertTrue(new_width % 16 == 0)
        self.assertTrue(new_height % 16 == 0)
        self.assertAlmostEqual(width / height, new_width / new_height, places=2)

    def test_custom_megapixels(self):
        width, height = 3840, 2160
        new_width, new_height = get_max_resolution(width, height, mega_pixels=1.0)
        self.assertLessEqual(new_width * new_height, 1.0 * 1000 * 1000)
        self.assertAlmostEqual(width / height, new_width / new_height, places=2)

    def test_custom_factor(self):
        width, height = 1920, 1080
        factor = 8
        new_width, new_height = get_max_resolution(width, height, factor=factor)
        self.assertTrue(new_width % factor == 0)
        self.assertTrue(new_height % factor == 0)
        
    def test_small_resolution_upscaling(self):
        width, height = 100, 50
        new_width, new_height = get_max_resolution(width, height)
        self.assertLessEqual(new_width * new_height, 0.25 * 1000 * 1000)
        self.assertAlmostEqual(width / height, new_width / new_height, places=2)

    def test_zero_height_raises_error(self):
        with self.assertRaises(ZeroDivisionError):
            get_max_resolution(1920, 0)
            
    def test_large_input_downscaling(self):
        width, height = 8000, 6000
        new_width, new_height = get_max_resolution(width, height, mega_pixels=0.5)
        self.assertLessEqual(new_width * new_height, 0.5 * 1000 * 1000)
        self.assertTrue(new_width < width)
        self.assertTrue(new_height < height)
        self.assertAlmostEqual(width / height, new_width / new_height, places=2)

if __name__ == '__main__':
    unittest.main()
