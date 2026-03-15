import unittest

import numpy as np

from definers.data import (
    TrainingData,
    order_dataset,
    prepare_data,
    split_dataset,
)


class TestPrepareDataHelpers(unittest.TestCase):
    def test_order_dataset_callable(self):

        arr = np.array([[3], [1], [2]])

        class SimpleDS:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        simple = SimpleDS(arr.tolist())

        ordered = order_dataset(simple, order_by=lambda x: x[0])
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0], [1])
        self.assertEqual(ordered[1], [2])
        self.assertEqual(ordered[2], [3])

    def test_split_dataset_basic(self):

        class SimpleDS:
            def __init__(self, n):
                self._n = n

            def __len__(self):
                return self._n

            def __getitem__(self, idx):
                return idx

        ds = SimpleDS(10)
        td = split_dataset(ds, val_frac=0.2, test_frac=0.1, batch_size=2)
        self.assertIsInstance(td, TrainingData)
        self.assertEqual(
            len(td.train.dataset)
            + (len(td.val.dataset) if td.val else 0)
            + (len(td.test.dataset) if td.test else 0),
            10,
        )

        self.assertTrue(len(td.train) > 0)

    def test_prepare_data_end_to_end(self):

        import os
        import tempfile

        tmp_files = []
        try:
            for value in [1, 2, 3]:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False, mode="w", newline=""
                )
                tmp.write(str(value))
                tmp.close()
                tmp_files.append(tmp.name)
            td = prepare_data(features=tmp_files, batch_size=1)
            self.assertIsInstance(td, TrainingData)

            self.assertEqual(len(td.train.dataset), 3)
        finally:
            for path in tmp_files:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def test_prepare_data_stratify(self):

        from unittest.mock import patch

        with patch("definers.data.fetch_dataset") as mf:
            mf.return_value = [
                {"x": 1, "label": 0},
                {"x": 2, "label": 1},
                {"x": 3, "label": 0},
                {"x": 4, "label": 1},
            ]
            td = prepare_data(
                remote_src="whatever",
                stratify="label",
                val_frac=0.2,
                test_frac=0.2,
            )
            self.assertIsInstance(td, TrainingData)
            total = (
                len(td.train.dataset)
                + (len(td.val.dataset) if td.val else 0)
                + (len(td.test.dataset) if td.test else 0)
            )
            self.assertEqual(total, 4)

            self.assertEqual(td.metadata.get("stratify"), "label")
            self.assertAlmostEqual(td.metadata.get("val_frac"), 0.2)
            self.assertAlmostEqual(td.metadata.get("test_frac"), 0.2)

    def test_prepare_data_caching(self):
        from unittest.mock import patch

        calls = []

        def fake_load_source(*args, **kwargs):
            calls.append(1)
            return [1, 2, 3]

        with patch("definers.data.load_source", side_effect=fake_load_source):
            td1 = prepare_data(features=["a"], batch_size=1)
            td2 = prepare_data(features=["a"], batch_size=1)
        self.assertIs(td1, td2)

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
