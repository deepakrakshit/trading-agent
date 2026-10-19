"""Small transactional SQLite repository for paper-trading audit records."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping


class PersistenceError(RuntimeError):
    """Raised when an idempotent record conflicts with existing data."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _json_payload(payload: Mapping[str, Any] | Any) -> str:
    return json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))


class SQLiteRepository:
    """SQLite-backed append-only records with idempotent natural keys.

    The repository stores structured JSON payloads alongside indexed columns;
    this keeps the first foundation compact while allowing future reporting to
    query orders/fills/transitions without replaying every event.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        """Close the underlying connection."""

        self.connection.close()

    def __enter__(self) -> "SQLiteRepository":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_id TEXT,
                idempotency_key TEXT,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(event_type, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS account_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transitions (
                transition_id TEXT PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def append_event(
        self,
        event_id: str,
        event_type: str,
        payload: Mapping[str, Any] | Any,
        *,
        aggregate_id: str | None = None,
        occurred_at: datetime | str | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        """Append an event, returning ``False`` for an identical replay."""

        if not event_id.strip() or not event_type.strip():
            raise ValueError("event_id and event_type must be non-empty")
        timestamp = occurred_at.isoformat() if isinstance(occurred_at, datetime) else str(occurred_at or "")
        encoded = _json_payload(payload)
        try:
            self.connection.execute(
                "INSERT INTO events(event_id,event_type,aggregate_id,idempotency_key,occurred_at,payload) "
                "VALUES(?,?,?,?,?,?)",
                (event_id, event_type, aggregate_id, idempotency_key, timestamp, encoded),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            existing = self.connection.execute(
                "SELECT payload FROM events WHERE event_id = ? OR (event_type = ? AND idempotency_key = ?)",
                (event_id, event_type, idempotency_key),
            ).fetchone()
            if existing is not None and existing["payload"] == encoded:
                return False
            raise PersistenceError("conflicting duplicate event") from exc

    def record_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        submitted_at: datetime,
        status: str,
        payload: Mapping[str, Any] | Any,
    ) -> bool:
        """Persist an order once; identical retries are no-ops."""

        encoded = _json_payload(payload)
        values = (client_order_id, symbol, side, quantity, submitted_at.isoformat(), status, encoded)
        try:
            self.connection.execute(
                "INSERT INTO orders(client_order_id,symbol,side,quantity,submitted_at,status,payload) VALUES(?,?,?,?,?,?,?)",
                values,
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            existing = self.get_order(client_order_id)
            if existing and existing["payload"] == encoded:
                return False
            raise PersistenceError(f"conflicting duplicate order: {client_order_id}") from exc

    def update_order_status(self, client_order_id: str, status: str) -> None:
        """Update only the indexed lifecycle status of an existing order."""

        cursor = self.connection.execute(
            "UPDATE orders SET status = ? WHERE client_order_id = ?", (status, client_order_id)
        )
        if cursor.rowcount != 1:
            raise PersistenceError(f"unknown order: {client_order_id}")
        self.connection.commit()

    def record_fill(
        self,
        *,
        fill_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        occurred_at: datetime,
        payload: Mapping[str, Any] | Any,
    ) -> bool:
        """Persist one fill exactly once, including duplicate callback safety."""

        encoded = _json_payload(payload)
        try:
            self.connection.execute(
                "INSERT INTO fills(fill_id,client_order_id,symbol,side,quantity,occurred_at,payload) VALUES(?,?,?,?,?,?,?)",
                (fill_id, client_order_id, symbol, side, quantity, occurred_at.isoformat(), encoded),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            existing = self.get_fill(fill_id)
            if existing and existing["payload"] == encoded:
                return False
            raise PersistenceError(f"conflicting duplicate fill: {fill_id}") from exc

    def record_snapshot(self, payload: Mapping[str, Any], *, occurred_at: datetime | None = None) -> int:
        """Persist an account snapshot and return its row ID."""

        cursor = self.connection.execute(
            "INSERT INTO account_snapshots(occurred_at,payload) VALUES(?,?)",
            (occurred_at.isoformat() if occurred_at else None, _json_payload(payload)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_transition(
        self,
        transition_id: str,
        payload: Mapping[str, Any],
        *,
        occurred_at: datetime,
    ) -> bool:
        """Store an RL transition with a stable idempotency key."""

        encoded = _json_payload(payload)
        try:
            self.connection.execute(
                "INSERT INTO transitions(transition_id,occurred_at,payload) VALUES(?,?,?)",
                (transition_id, occurred_at.isoformat(), encoded),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            existing = self.get_transition(transition_id)
            if existing and existing["payload"] == encoded:
                return False
            raise PersistenceError(f"conflicting duplicate transition: {transition_id}") from exc

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_fill(self, fill_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM fills WHERE fill_id = ?", (fill_id,)).fetchone()
        return dict(row) if row else None

    def get_transition(self, transition_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM transitions WHERE transition_id = ?", (transition_id,)
        ).fetchone()
        return dict(row) if row else None

    def events(self, *, event_type: str | None = None) -> tuple[dict[str, Any], ...]:
        """Return event rows in append order."""

        if event_type is None:
            rows = self.connection.execute("SELECT * FROM events ORDER BY rowid").fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY rowid", (event_type,)
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def latest_snapshot(self) -> dict[str, Any] | None:
        """Return the most recently stored account snapshot payload."""

        row = self.connection.execute(
            "SELECT * FROM account_snapshots ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
