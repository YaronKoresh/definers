from definers.persistence import database as database_module
from definers.persistence.database import Database


def test_history_applies_days_filter_and_returns_descending_records(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database_module, "time", lambda: 1_000_000)
    database = Database(str(tmp_path))

    database.push("items", {"id": "1", "status": "active"}, timestamp=913_599)
    database.push("items", {"id": "2", "status": "active"}, timestamp=913_600)
    database.push("items", {"id": "3", "status": "active"}, timestamp=920_000)
    database.push("items", {"id": "4", "status": "inactive"}, timestamp=930_000)

    result = database.history("items", filters={"status": "active"}, days=1)

    assert result == [
        {"id": "3", "status": "active"},
        {"id": "2", "status": "active"},
    ]


def test_latest_uses_custom_identifier_and_filters_after_deduplication(tmp_path):
    database = Database(str(tmp_path))

    database.push(
        "items",
        {"slug": "alpha", "published": "true", "value": "old"},
        timestamp=1_000,
    )
    database.push(
        "items",
        {"slug": "alpha", "published": "false", "value": "new"},
        timestamp=2_000,
    )
    database.push(
        "items",
        {"slug": "beta", "published": "true", "value": "latest"},
        timestamp=1_500,
    )

    result = database.latest(
        "items",
        filters={"published": "true"},
        identifierKey="slug",
    )

    assert result == [{"slug": "beta", "published": "true", "value": "latest"}]


def test_latest_returns_mapping_for_selected_databases(tmp_path):
    database = Database(str(tmp_path))

    database.push("items", {"id": "1", "value": "one"}, timestamp=1_000)
    database.push("events", {"id": "2", "value": "two"}, timestamp=2_000)

    result = database.latest(["items", "events"])

    assert result == {
        "items": [{"id": "1", "value": "one"}],
        "events": [{"id": "2", "value": "two"}],
    }