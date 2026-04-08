import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any
from uuid import uuid4

SECONDS_PER_DAY = 86400


@dataclass(frozen=True, slots=True)
class DatabaseRecord:
    timestamp: int
    data: dict[str, str]

    def as_history_item(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp),
            "data": self.data,
        }


class Database:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def _database_path(self, db: str) -> str:
        return os.path.join(self.path, db)

    def _record_path(self, db: str, timestamp: int) -> str:
        return os.path.join(self._database_path(db), str(timestamp))

    def _parse_record_timestamp(self, directory_name: str) -> int | None:
        timestamp_token = directory_name.split("-", 1)[0].strip()
        if not timestamp_token:
            return None
        try:
            return int(timestamp_token)
        except ValueError:
            return None

    def _create_record_path(self, db: str, timestamp: int) -> str:
        db_path = self._database_path(db)
        os.makedirs(db_path, exist_ok=True)
        primary_path = self._record_path(db, timestamp)
        try:
            os.mkdir(primary_path)
            return primary_path
        except FileExistsError:
            pass

        while True:
            candidate_path = os.path.join(
                db_path,
                f"{timestamp}-{uuid4().hex[:12]}",
            )
            try:
                os.mkdir(candidate_path)
                return candidate_path
            except FileExistsError:
                continue

    def _history_start_timestamp(self, days: int | float | None) -> float:
        if days is None or not isinstance(days, (int, float)):
            return 0
        return time() - days * SECONDS_PER_DAY

    def _normalize_timestamp(self, timestamp: int | None) -> int:
        if timestamp is None:
            return int(time())
        if isinstance(timestamp, int):
            return timestamp
        try:
            return int(timestamp)
        except (ValueError, TypeError):
            return int(time())

    def _list_database_names(self) -> list[str]:
        return [
            name
            for name in os.listdir(self.path)
            if os.path.isdir(self._database_path(name))
        ]

    def _list_record_timestamps(
        self,
        db_path: str,
        start_timestamp: float,
    ) -> list[tuple[int, str]]:
        try:
            timestamps: list[tuple[int, str]] = []
            for directory_name in os.listdir(db_path):
                timestamp = self._parse_record_timestamp(directory_name)
                if timestamp is None:
                    continue
                if timestamp >= start_timestamp:
                    timestamps.append((timestamp, directory_name))
            return sorted(
                timestamps,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
        except FileNotFoundError:
            return []

    def _read_record(self, record_path: str) -> dict[str, str]:
        item_data: dict[str, str] = {}
        for key_file in os.listdir(record_path):
            with open(
                os.path.join(record_path, key_file), encoding="utf-8"
            ) as file_handle:
                item_data[key_file] = file_handle.read()
        return item_data

    def _matches_filters(
        self,
        item_data: dict[str, str],
        filters: dict[str, Any],
    ) -> bool:
        for key, value in filters.items():
            if item_data.get(key) != str(value):
                return False
        return True

    def _latest_items_by_identifier(
        self,
        full_history: list[DatabaseRecord],
        identifier_key: str,
    ) -> dict[str, DatabaseRecord]:
        latest_items: dict[str, DatabaseRecord] = {}
        for item in full_history:
            item_id = item.data.get(identifier_key)
            if item_id is None:
                continue
            if (
                item_id not in latest_items
                or item.timestamp > latest_items[item_id].timestamp
            ):
                latest_items[item_id] = item
        return latest_items

    def _get_history(
        self,
        db: str,
        filters: dict[str, Any] | None = None,
        days: int | float | None = None,
    ) -> list[DatabaseRecord]:
        filters = filters or {}
        db_path = self._database_path(db)
        if not os.path.exists(db_path):
            return []
        start_timestamp = self._history_start_timestamp(days)
        timestamp_dirs = self._list_record_timestamps(db_path, start_timestamp)
        results: list[DatabaseRecord] = []
        for timestamp_value, directory_name in timestamp_dirs:
            record_path = os.path.join(db_path, directory_name)
            if not os.path.isdir(record_path):
                continue
            item_data = self._read_record(record_path)
            if self._matches_filters(item_data, filters):
                results.append(DatabaseRecord(timestamp_value, item_data))
        return results

    def history(
        self,
        db: str,
        filters: dict[str, Any] | None = None,
        days: int | float | None = None,
    ) -> list[dict[str, str]]:
        full_history = self._get_history(db, filters, days)
        return [item.data for item in full_history]

    def push(
        self,
        db: str,
        data: dict[str, Any],
        timestamp: int | None = None,
    ) -> None:
        normalized_timestamp = self._normalize_timestamp(timestamp)
        record_path = self._create_record_path(db, normalized_timestamp)
        for key, value in data.items():
            file_path = os.path.join(record_path, key)
            with open(file_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(str(value))

    def latest(
        self,
        db: str | list[str] = "*",
        filters: dict[str, Any] | None = None,
        days: int | float | None = None,
        identifierKey: str = "id",
    ) -> list[dict[str, str]] | dict[str, Any]:
        filters = filters or {}
        if db == "*":
            return {
                db_name: self.latest(db_name, filters, days, identifierKey)
                for db_name in self._list_database_names()
            }
        if isinstance(db, list):
            return {
                db_name: self.latest(db_name, filters, days, identifierKey)
                for db_name in db
            }
        full_history = self._get_history(db, days=days)
        latest_items = self._latest_items_by_identifier(
            full_history, identifierKey
        )
        filtered_results = list(latest_items.values())
        if filters:
            filtered_results = [
                item
                for item in filtered_results
                if self._matches_filters(item.data, filters)
            ]
        sorted_results = sorted(
            filtered_results,
            key=lambda item: item.timestamp,
            reverse=True,
        )
        return [item.data for item in sorted_results]

    def clean(
        self, db: str | list[str] = "*", identifierKey: str = "id"
    ) -> None:
        if db == "*":
            for db_name in self._list_database_names():
                self.clean(db_name, identifierKey)
            return
        if isinstance(db, list):
            for db_name in db:
                self.clean(db_name, identifierKey)
            return
        full_history = self._get_history(db)
        latest_items = self._latest_items_by_identifier(
            full_history, identifierKey
        )
        db_path = self._database_path(db)
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        for item in latest_items.values():
            self.push(db, item.data, item.timestamp)
