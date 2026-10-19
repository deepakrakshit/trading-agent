"""Durable SQLite event and paper-account records."""

from .sqlite import PersistenceError, SQLiteRepository

__all__ = ["PersistenceError", "SQLiteRepository"]
