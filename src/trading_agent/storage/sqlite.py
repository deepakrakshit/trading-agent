"""Small transactional SQLite repository for paper-trading audit records."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping

from ..data.bars import utc_timestamp


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
    return json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        raise ValueError("an explicit timezone-aware event timestamp is required")
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return utc_timestamp(parsed).isoformat()


class SQLiteRepository:
    """SQLite-backed append-only records with idempotent natural keys.

    The repository stores structured JSON payloads alongside indexed columns;
    this keeps the first foundation compact while allowing future reporting to
    query orders/fills/transitions without replaying every event.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            if not Path(self.path).parent.is_dir():
                raise ValueError("SQLite parent directory must already exist")
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._transaction_depth = 0
        try:
            self._create_schema()
            integrity = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise PersistenceError(f"SQLite integrity check failed: {integrity}")
        except BaseException:
            self.connection.close()
            raise

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
                quantity INTEGER NOT NULL CHECK(typeof(quantity) = 'integer' AND quantity > 0),
                submitted_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(typeof(quantity) = 'integer' AND quantity > 0),
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(client_order_id) REFERENCES orders(client_order_id)
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
            CREATE TABLE IF NOT EXISTS states (
                state_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit a group of records atomically; nested groups use savepoints."""

        depth = self._transaction_depth
        savepoint = f"repository_{depth}"
        self.connection.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield
            if depth == 0:
                self.connection.commit()
            else:
                self.connection.execute(f"RELEASE {savepoint}")
        except BaseException:
            if depth == 0:
                self.connection.rollback()
            else:
                self.connection.execute(f"ROLLBACK TO {savepoint}")
                self.connection.execute(f"RELEASE {savepoint}")
            raise
        finally:
            self._transaction_depth -= 1

    def _commit(self) -> None:
        if self._transaction_depth == 0:
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
        timestamp = _timestamp(occurred_at)
        encoded = _json_payload(payload)
        try:
            self.connection.execute(
                "INSERT INTO events(event_id,event_type,aggregate_id,idempotency_key,occurred_at,payload) "
                "VALUES(?,?,?,?,?,?)",
                (event_id, event_type, aggregate_id, idempotency_key, timestamp, encoded),
            )
            self._commit()
            return True
        except sqlite3.IntegrityError as exc:
            if self._transaction_depth == 0:
                self.connection.rollback()
            existing = self.connection.execute(
                "SELECT * FROM events WHERE event_id = ? OR (event_type = ? AND idempotency_key = ?)",
                (event_id, event_type, idempotency_key),
            ).fetchone()
            if (
                existing is not None
                and existing["payload"] == encoded
                and existing["event_type"] == event_type
                and existing["aggregate_id"] == aggregate_id
                and existing["occurred_at"] == timestamp
                and existing["idempotency_key"] == idempotency_key
            ):
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
        values = (client_order_id, symbol, side, quantity, _timestamp(submitted_at), status, encoded)
        try:
            self.connection.execute(
                "INSERT INTO orders(client_order_id,symbol,side,quantity,submitted_at,status,payload) VALUES(?,?,?,?,?,?,?)",
                values,
            )
            self._commit()
            return True
        except sqlite3.IntegrityError as exc:
            if self._transaction_depth == 0:
                self.connection.rollback()
            existing = self.get_order(client_order_id)
            if existing and tuple(existing[name] for name in ("client_order_id", "symbol", "side", "quantity", "submitted_at", "status", "payload")) == values:
                return False
            raise PersistenceError(f"conflicting duplicate order: {client_order_id}") from exc

    def update_order_status(self, client_order_id: str, status: str) -> None:
        """Update only the indexed lifecycle status of an existing order."""

        cursor = self.connection.execute(
            "UPDATE orders SET status = ? WHERE client_order_id = ?", (status, client_order_id)
        )
        if cursor.rowcount != 1:
            if self._transaction_depth == 0:
                self.connection.rollback()
            raise PersistenceError(f"unknown order: {client_order_id}")
        self._commit()

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
        timestamp = _timestamp(occurred_at)
        try:
            self.connection.execute(
                "INSERT INTO fills(fill_id,client_order_id,symbol,side,quantity,occurred_at,payload) VALUES(?,?,?,?,?,?,?)",
                (fill_id, client_order_id, symbol, side, quantity, timestamp, encoded),
            )
            self._commit()
            return True
        except sqlite3.IntegrityError as exc:
            if self._transaction_depth == 0:
                self.connection.rollback()
            existing = self.get_fill(fill_id)
            if existing and (
                existing["payload"] == encoded
                and existing["client_order_id"] == client_order_id
                and existing["symbol"] == symbol
                and existing["side"] == side
                and existing["quantity"] == quantity
                and existing["occurred_at"] == timestamp
            ):
                return False
            raise PersistenceError(f"conflicting duplicate fill: {fill_id}") from exc

    def record_snapshot(self, payload: Mapping[str, Any], *, occurred_at: datetime | None = None) -> int:
        """Persist an account snapshot and return its row ID."""

        cursor = self.connection.execute(
            "INSERT INTO account_snapshots(occurred_at,payload) VALUES(?,?)",
            (_timestamp(occurred_at) if occurred_at else None, _json_payload(payload)),
        )
        self._commit()
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
        timestamp = _timestamp(occurred_at)
        try:
            self.connection.execute(
                "INSERT INTO transitions(transition_id,occurred_at,payload) VALUES(?,?,?)",
                (transition_id, timestamp, encoded),
            )
            self._commit()
            return True
        except sqlite3.IntegrityError as exc:
            if self._transaction_depth == 0:
                self.connection.rollback()
            existing = self.get_transition(transition_id)
            if existing and existing["payload"] == encoded and existing["occurred_at"] == timestamp:
                return False
            raise PersistenceError(f"conflicting duplicate transition: {transition_id}") from exc

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        """Read an order row including its current lifecycle status."""

        row = self.connection.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_fill(self, fill_id: str) -> dict[str, Any] | None:
        """Read one immutable fill row."""

        row = self.connection.execute("SELECT * FROM fills WHERE fill_id = ?", (fill_id,)).fetchone()
        return dict(row) if row else None

    def get_transition(self, transition_id: str) -> dict[str, Any] | None:
        """Read one immutable numerical replay transition."""

        row = self.connection.execute(
            "SELECT * FROM transitions WHERE transition_id = ?", (transition_id,)
        ).fetchone()
        return dict(row) if row else None

    def transitions(self) -> tuple[dict[str, Any], ...]:
        """Read durable replay transitions in original commit order."""

        rows = self.connection.execute("SELECT * FROM transitions ORDER BY rowid").fetchall()
        return tuple(dict(row) for row in rows)

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

    def save_state(self, state_key: str, payload: Mapping[str, Any]) -> None:
        """Upsert a restart checkpoint in the caller's transaction."""

        if not state_key.strip():
            raise ValueError("state_key must be non-empty")
        self.connection.execute(
            "INSERT INTO states(state_key,payload) VALUES(?,?) "
            "ON CONFLICT(state_key) DO UPDATE SET payload=excluded.payload",
            (state_key, _json_payload(payload)),
        )
        self._commit()

    def load_state(self, state_key: str) -> dict[str, Any] | None:
        """Read a decoded restart checkpoint."""

        row = self.connection.execute("SELECT payload FROM states WHERE state_key=?", (state_key,)).fetchone()
        return json.loads(row["payload"]) if row else None
