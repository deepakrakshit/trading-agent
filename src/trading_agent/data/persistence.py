"""Immutable CSV/Parquet market snapshots with checked source manifests.

Writes never replace either an existing data file or its JSON sidecar.  Data
and manifest are fully staged before no-clobber publication; interrupted pairs
fail closed on load. Decimal values and UTC bar-start timestamps round-trip
losslessly. Parquet needs optional ``pyarrow``; CSV uses only the stdlib.

The manifest contains source facts only, through an explicit metadata
allowlist. Never pass credentials as metadata. SHA-256 detects changed files;
pin ``manifest.digest`` externally and pass ``expected_manifest_sha256`` when
loading to also detect replacement of both the file and its sidecar. Hashes
alone do not authenticate a vendor or establish data licensing/entitlements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from .bars import DataProvenance, DataValidationError, MarketBar, utc_timestamp
from .validation import validate_bars

_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume", "source", "complete")
_SOURCE_FIELDS = frozenset({
    "provider", "exchange", "interval", "dataset_version", "instrument_mapping_version",
    "calendar_version", "timestamp_convention", "api_origin", "endpoint", "request_start",
    "request_end", "license_reference",
})
_BAR_INTERVALS = {"FIVE_MINUTE": timedelta(minutes=5)}


class SnapshotError(DataValidationError):
    """A snapshot or manifest cannot be trusted as a coherent data artifact."""


class SnapshotDependencyError(ImportError):
    """Parquet was requested without the optional pyarrow dependency."""


def _source_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise SnapshotError("source must be a non-empty source identifier")
    return value


def _metadata(values: Mapping[str, str]) -> Mapping[str, str]:
    if set(values) - _SOURCE_FIELDS:
        raise SnapshotError("source metadata contains a non-allowlisted field")
    clean = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 1024 or any(ord(char) < 32 for char in value):
            raise SnapshotError("source metadata values must be short printable strings")
        if key in ("api_origin", "license_reference") and "://" in value:
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise SnapshotError("source metadata URLs cannot contain credentials, queries or fragments")
            if key == "api_origin" and parsed.path not in ("", "/"):
                raise SnapshotError("api_origin must be an origin, not a request URL")
        if key == "endpoint" and (not value.startswith("/") or any(char in value for char in ("?", "#", "\\"))):
            raise SnapshotError("endpoint metadata must be a path without query data")
        clean[key] = value
    return MappingProxyType(clean)


def _validate_bar_contract(metadata: Mapping[str, str]) -> timedelta:
    if metadata.get("timestamp_convention") != "BAR_START":
        raise SnapshotError("snapshot metadata must declare timestamp_convention=BAR_START")
    interval_name = metadata.get("interval")
    if interval_name not in _BAR_INTERVALS:
        raise SnapshotError("snapshot metadata must declare interval=FIVE_MINUTE")
    return _BAR_INTERVALS[interval_name]


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".manifest.json")


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    file_name: str
    format: str
    source: str
    provenance: str
    row_count: int
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    captured_at: datetime
    data_sha256: str
    source_metadata: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool) or self.format not in ("csv", "parquet"):
            raise SnapshotError("unsupported snapshot schema/format")
        if not isinstance(self.file_name, str) or Path(self.file_name).name != self.file_name or self.file_name in ("", ".", ".."):
            raise SnapshotError("invalid snapshot file name")
        _source_name(self.source)
        try:
            provenance = DataProvenance(self.provenance).value
        except (TypeError, ValueError):
            raise SnapshotError("unrecognized snapshot provenance") from None
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count <= 0:
            raise SnapshotError("snapshot row_count must be positive")
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols) or tuple(sorted(set(symbols))) != symbols:
            raise SnapshotError("snapshot symbols must be non-empty, unique and sorted")
        start, end, captured = (utc_timestamp(value) for value in (self.start, self.end, self.captured_at))
        if end < start or captured < end:
            raise SnapshotError("snapshot date range/capture time is inconsistent")
        if not isinstance(self.data_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.data_sha256):
            raise SnapshotError("snapshot data hash must be SHA-256")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "captured_at", captured)
        metadata = _metadata(self.source_metadata)
        _validate_bar_contract(metadata)
        object.__setattr__(self, "source_metadata", metadata)

    def to_mapping(self) -> dict[str, Any]:
        """``start`` and ``end`` are the first/last actual bar starts, inclusive."""

        return {
            "schema_version": self.schema_version, "file_name": self.file_name,
            "format": self.format, "source": self.source, "provenance": self.provenance,
            "row_count": self.row_count, "symbols": list(self.symbols),
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "captured_at": self.captured_at.isoformat(), "data_sha256": self.data_sha256,
            "source_metadata": dict(self.source_metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    @property
    def digest(self) -> str:
        return _hash(self.to_json().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    bars: tuple[MarketBar, ...]
    manifest: SnapshotManifest


def _parquet_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise SnapshotDependencyError("Parquet snapshots require pyarrow; CSV remains available") from None
    return pa, pq


def _encode(bars: tuple[MarketBar, ...], format_: str) -> bytes:
    rows = [bar.to_mapping() for bar in bars]
    if format_ == "csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "complete": "true"})
        return stream.getvalue().encode("utf-8")
    pa, pq = _parquet_modules()
    schema = pa.schema([(name, pa.bool_() if name == "complete" else pa.string()) for name in _COLUMNS])
    table = pa.Table.from_pylist(rows, schema=schema)
    stream = pa.BufferOutputStream()
    pq.write_table(table, stream, compression="zstd")
    return stream.getvalue().to_pybytes()


def _decode(payload: bytes, format_: str) -> tuple[MarketBar, ...]:
    try:
        if format_ == "csv":
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
            if tuple(reader.fieldnames or ()) != _COLUMNS:
                raise SnapshotError("snapshot columns do not match the canonical schema")
            rows = list(reader)
            for row in rows:
                if row.get("complete") != "true" or set(row) != set(_COLUMNS):
                    raise SnapshotError("snapshot contains malformed or incomplete rows")
                row["complete"] = True
        else:
            pa, pq = _parquet_modules()
            table = pq.read_table(pa.BufferReader(payload))
            if tuple(table.column_names) != _COLUMNS:
                raise SnapshotError("snapshot columns do not match the canonical schema")
            rows = table.to_pylist()
        return validate_bars(MarketBar.from_mapping(row) for row in rows)
    except SnapshotDependencyError:
        raise
    except Exception:
        raise SnapshotError("snapshot rows failed schema/ordering validation") from None


def _format(path: Path) -> str:
    if path.suffix not in (".csv", ".parquet"):
        raise SnapshotError("snapshot path must end in .csv or .parquet")
    return path.suffix[1:]


def _publish(path: Path, data: bytes, sidecar: bytes) -> None:
    staged, published = [], []
    try:
        for payload in (data, sidecar):
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".snapshot-", delete=False) as handle:
                staged.append(Path(handle.name))
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in zip(staged, (path, manifest_path(path))):
            # A hard link atomically publishes a complete file without ever
            # replacing an existing destination (unlike os.replace).
            os.link(temporary, destination)
            published.append(destination)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def write_snapshot(
    path: str | Path,
    bars: Iterable[MarketBar],
    *,
    source: str,
    provenance: str | DataProvenance,
    source_metadata: Mapping[str, str] | None = None,
    captured_at: datetime | None = None,
) -> SnapshotManifest:
    path = Path(path)
    format_ = _format(path)
    if not path.parent.is_dir():
        raise FileNotFoundError("snapshot parent directory must already exist")
    if path.exists() or manifest_path(path).exists():
        raise FileExistsError("immutable snapshot or manifest already exists")
    captured = utc_timestamp(captured_at) if captured_at is not None else datetime.now(timezone.utc)
    materialised = validate_bars(bars, as_of=captured)
    if not materialised:
        raise SnapshotError("empty market snapshots are not accepted")
    try:
        provenance = DataProvenance(provenance).value
    except (TypeError, ValueError):
        raise SnapshotError("unrecognized snapshot provenance") from None
    if any(bar.source != provenance for bar in materialised):
        raise SnapshotError("bar source does not match declared snapshot provenance")
    # Validate source fields before serialising data or touching the filesystem.
    source = _source_name(source)
    metadata_values = dict(source_metadata or {})
    # The canonical project snapshot format is five-minute BAR_START data.
    # Keep direct callers backwards-compatible while rejecting any explicit
    # contract that disagrees with that format.
    metadata_values.setdefault("timestamp_convention", "BAR_START")
    metadata_values.setdefault("interval", "FIVE_MINUTE")
    metadata = _metadata(metadata_values)
    interval = _validate_bar_contract(metadata)
    if any(bar.timestamp + interval > captured for bar in materialised):
        raise SnapshotError("snapshot contains a forming/unavailable candle at capture time")
    payload = _encode(materialised, format_)
    manifest = SnapshotManifest(
        path.name, format_, source, provenance, len(materialised),
        tuple(sorted({bar.symbol for bar in materialised})),
        min(bar.timestamp for bar in materialised), max(bar.timestamp for bar in materialised),
        captured, _hash(payload), metadata,
    )
    _publish(path, payload, manifest.to_json().encode("utf-8"))
    return manifest


def _manifest(payload: bytes) -> SnapshotManifest:
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotError("duplicate manifest field")
            result[key] = value
        return result

    try:
        data = json.loads(payload, object_pairs_hook=unique_fields)
        for name in ("start", "end", "captured_at"):
            data[name] = datetime.fromisoformat(data[name])
        return SnapshotManifest(**data)
    except Exception:
        raise SnapshotError("invalid snapshot manifest") from None


def load_snapshot(path: str | Path, *, expected_manifest_sha256: str | None = None) -> MarketSnapshot:
    path = Path(path)
    format_ = _format(path)
    try:
        with manifest_path(path).open("rb") as handle:
            manifest_bytes = handle.read(1024 * 1024 + 1)
        if len(manifest_bytes) > 1024 * 1024:
            raise SnapshotError("snapshot manifest exceeds its size bound")
        payload = path.read_bytes()
    except OSError:
        raise SnapshotError("snapshot data/manifest pair is unavailable") from None
    if expected_manifest_sha256 is not None:
        if not isinstance(expected_manifest_sha256, str) or not hmac.compare_digest(_hash(manifest_bytes), expected_manifest_sha256):
            raise SnapshotError("snapshot manifest hash mismatch")
    manifest = _manifest(manifest_bytes)
    if manifest.file_name != path.name or manifest.format != format_:
        raise SnapshotError("snapshot file name/format does not match manifest")
    if not hmac.compare_digest(_hash(payload), manifest.data_sha256):
        raise SnapshotError("snapshot data hash mismatch")
    bars = _decode(payload, format_)
    if (
        len(bars) != manifest.row_count
        or tuple(sorted({bar.symbol for bar in bars})) != manifest.symbols
        or any(bar.source != manifest.provenance for bar in bars)
        or not bars
        or min(bar.timestamp for bar in bars) != manifest.start
        or max(bar.timestamp for bar in bars) != manifest.end
    ):
        raise SnapshotError("snapshot records do not match manifest claims")
    return MarketSnapshot(bars, manifest)


__all__ = ["MarketSnapshot", "SnapshotDependencyError", "SnapshotError", "SnapshotManifest", "load_snapshot", "manifest_path", "write_snapshot"]
