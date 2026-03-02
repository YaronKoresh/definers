import unittest

from src.definers._web import geo_new_york


class TestGeoNewYork(unittest.TestCase):
    def test_returns_dict_with_latitude_and_longitude(self) -> None:
        result = geo_new_york()
        self.assertIsInstance(result, dict)
        self.assertIn("latitude", result)
        self.assertIn("longitude", result)

    def test_latitude_in_new_york_range(self) -> None:
        result = geo_new_york()
        self.assertGreaterEqual(result["latitude"], 40.5)
        self.assertLessEqual(result["latitude"], 40.9)

    def test_longitude_in_new_york_range(self) -> None:
        result = geo_new_york()
        self.assertGreaterEqual(result["longitude"], -74.2)
        self.assertLessEqual(result["longitude"], -73.7)

    def test_returns_float_values(self) -> None:
        result = geo_new_york()
        self.assertIsInstance(result["latitude"], float)
        self.assertIsInstance(result["longitude"], float)

    def test_returns_different_values_across_calls(self) -> None:
        results = [geo_new_york() for _ in range(50)]
        latitudes = {r["latitude"] for r in results}
        self.assertGreater(len(latitudes), 1)


if __name__ == "__main__":
    unittest.main()
