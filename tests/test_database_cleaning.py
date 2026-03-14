import os

from definers.persistence.database import Database


def test_clean_keeps_latest_record_per_identifier_and_preserves_timestamps(tmp_path):
    database = Database(str(tmp_path))

    database.push("items", {"id": "1", "value": "old"}, timestamp=1_000)
    database.push("items", {"id": "1", "value": "new"}, timestamp=2_000)
    database.push("items", {"id": "2", "value": "other"}, timestamp=1_500)

    database.clean("items")

    assert sorted(os.listdir(tmp_path / "items")) == ["1500", "2000"]
    assert database.history("items") == [
        {"id": "1", "value": "new"},
        {"id": "2", "value": "other"},
    ]


def test_clean_uses_custom_identifier_key(tmp_path):
    database = Database(str(tmp_path))

    database.push("items", {"slug": "alpha", "value": "old"}, timestamp=1_000)
    database.push("items", {"slug": "alpha", "value": "new"}, timestamp=2_000)
    database.push("items", {"slug": "beta", "value": "keep"}, timestamp=1_500)

    database.clean("items", identifierKey="slug")

    assert database.history("items") == [
        {"slug": "alpha", "value": "new"},
        {"slug": "beta", "value": "keep"},
    ]


def test_clean_with_database_list_only_updates_requested_databases(tmp_path):
    database = Database(str(tmp_path))

    database.push("items", {"id": "1", "value": "old"}, timestamp=1_000)
    database.push("items", {"id": "1", "value": "new"}, timestamp=2_000)
    database.push("events", {"id": "7", "value": "old"}, timestamp=3_000)
    database.push("events", {"id": "7", "value": "new"}, timestamp=4_000)

    database.clean(["items"])

    assert database.history("items") == [{"id": "1", "value": "new"}]
    assert database.history("events") == [
        {"id": "7", "value": "new"},
        {"id": "7", "value": "old"},
    ]


def test_clean_wildcard_updates_all_databases(tmp_path):
    database = Database(str(tmp_path))

    database.push("items", {"id": "1", "value": "old"}, timestamp=1_000)
    database.push("items", {"id": "1", "value": "new"}, timestamp=2_000)
    database.push("events", {"id": "2", "value": "old"}, timestamp=3_000)
    database.push("events", {"id": "2", "value": "new"}, timestamp=4_000)

    database.clean("*")

    assert database.latest("*") == {
        "items": [{"id": "1", "value": "new"}],
        "events": [{"id": "2", "value": "new"}],
    }
    assert sorted(os.listdir(tmp_path / "items")) == ["2000"]
    assert sorted(os.listdir(tmp_path / "events")) == ["4000"]