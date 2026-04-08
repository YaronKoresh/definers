import tempfile
import unittest
from pathlib import Path

from definers.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(self._tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_push_and_history(self):
        self.db.push("items", {"id": "1", "name": "apple"}, timestamp=1000)
        result = self.db.history("items")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "apple")

    def test_history_empty_db(self):
        result = self.db.history("nonexistent")
        self.assertEqual(result, [])

    def test_push_multiple_records(self):
        self.db.push("items", {"id": "1", "value": "a"}, timestamp=1000)
        self.db.push("items", {"id": "2", "value": "b"}, timestamp=2000)
        result = self.db.history("items")
        self.assertEqual(len(result), 2)

    def test_history_filter(self):
        self.db.push("items", {"id": "1", "status": "active"}, timestamp=1000)
        self.db.push("items", {"id": "2", "status": "inactive"}, timestamp=2000)
        result = self.db.history("items", filters={"status": "active"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "active")

    def test_latest_returns_most_recent_per_id(self):
        self.db.push("items", {"id": "1", "value": "old"}, timestamp=1000)
        self.db.push("items", {"id": "1", "value": "new"}, timestamp=2000)
        result = self.db.latest("items")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], "new")

    def test_latest_multiple_ids(self):
        self.db.push("items", {"id": "a", "value": "1"}, timestamp=1000)
        self.db.push("items", {"id": "b", "value": "2"}, timestamp=2000)
        result = self.db.latest("items")
        self.assertEqual(len(result), 2)

    def test_clean_removes_old_duplicates(self):
        self.db.push("items", {"id": "1", "value": "old"}, timestamp=1000)
        self.db.push("items", {"id": "1", "value": "new"}, timestamp=2000)
        self.db.clean("items")
        result = self.db.history("items")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], "new")

    def test_push_auto_timestamp(self):
        self.db.push("items", {"id": "x", "v": "1"})
        result = self.db.history("items")
        self.assertEqual(len(result), 1)

    def test_latest_with_wildcard_returns_database_mapping(self):
        self.db.push("items", {"id": "1", "value": "one"}, timestamp=1000)
        self.db.push("events", {"id": "2", "value": "two"}, timestamp=2000)

        result = self.db.latest("*")

        self.assertEqual(result["items"][0]["value"], "one")
        self.assertEqual(result["events"][0]["value"], "two")

    def test_history_ignores_non_timestamp_directories(self):
        invalid_directory = Path(self._tmpdir) / "items" / "not-a-timestamp"
        invalid_directory.mkdir(parents=True)
        (invalid_directory / "id").write_text("ignored", encoding="utf-8")

        self.db.push("items", {"id": "1", "value": "ok"}, timestamp=1000)

        result = self.db.history("items")

        self.assertEqual(result, [{"id": "1", "value": "ok"}])

    def test_push_same_timestamp_keeps_distinct_records(self):
        self.db.push("items", {"id": "1", "value": "first"}, timestamp=1000)
        self.db.push("items", {"id": "2", "value": "second"}, timestamp=1000)

        record_directories = sorted(
            path.name for path in (Path(self._tmpdir) / "items").iterdir()
        )
        history = self.db.history("items")

        self.assertEqual(len(record_directories), 2)
        self.assertIn("1000", record_directories)
        self.assertTrue(
            any(
                directory_name.startswith("1000-")
                for directory_name in record_directories
                if directory_name != "1000"
            )
        )
        self.assertCountEqual(
            history,
            [
                {"id": "1", "value": "first"},
                {"id": "2", "value": "second"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
