from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from unittest.mock import patch

import pytest

from trading_agent.data import MarketBar, synthetic_bars
from trading_agent.data.persistence import SnapshotDependencyError, SnapshotError, load_snapshot, manifest_path, write_snapshot

START = datetime(2026, 1, 5, 3, 45, tzinfo=timezone.utc)
CAPTURED = START + timedelta(hours=1)


def bars():
    return tuple(MarketBar("ABC-EQ", START + timedelta(minutes=5 * index), Decimal("100.000001"), Decimal("101.123456789"), Decimal("99.999999"), Decimal("100.100000"), Decimal("123.4500"), source="HISTORICAL") for index in range(3))


def write(path, **kwargs):
    return write_snapshot(path, bars(), source="ANGEL_ONE_SMARTAPI", provenance="HISTORICAL", captured_at=CAPTURED, **kwargs)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_lossless_round_trip_and_manifest_source_range_hash(tmp_path, suffix):
    if suffix == "parquet":
        pytest.importorskip("pyarrow")
    path = tmp_path / f"candles.{suffix}"
    manifest = write(path, source_metadata={"provider": "ANGEL_ONE_SMARTAPI", "exchange": "NSE", "interval": "FIVE_MINUTE"})
    loaded = load_snapshot(path, expected_manifest_sha256=manifest.digest)
    assert loaded.bars == bars()
    assert loaded.manifest == manifest
    assert manifest.start == START and manifest.end == START + timedelta(minutes=10)
    assert manifest.source == "ANGEL_ONE_SMARTAPI" and manifest.provenance == "HISTORICAL"
    assert manifest.symbols == ("ABC-EQ",) and manifest.row_count == 3
    assert manifest.data_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.captured_at == CAPTURED
    assert json.loads(manifest_path(path).read_text())["source_metadata"]["interval"] == "FIVE_MINUTE"
    with pytest.raises(TypeError):
        manifest.source_metadata["token"] = "never-stored"


def test_csv_is_reproducible_for_same_bars_metadata_and_capture_time(tmp_path):
    first, second = tmp_path / "one" / "bars.csv", tmp_path / "two" / "bars.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    one, two = write(first), write(second)
    assert first.read_bytes() == second.read_bytes()
    assert one.digest == two.digest


def test_existing_data_or_manifest_is_never_overwritten(tmp_path):
    path = tmp_path / "bars.csv"
    write(path)
    original = path.read_bytes(), manifest_path(path).read_bytes()
    with pytest.raises(FileExistsError):
        write(path)
    assert original == (path.read_bytes(), manifest_path(path).read_bytes())
    path.unlink()
    with pytest.raises(FileExistsError):
        write(path)
    assert not path.exists()
    assert manifest_path(path).read_bytes() == original[1]
    with pytest.raises(SnapshotError, match="unavailable"):
        load_snapshot(path)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_tampered_data_fails_hash_before_decode(tmp_path, suffix):
    if suffix == "parquet":
        pytest.importorskip("pyarrow")
    path = tmp_path / f"bars.{suffix}"
    write(path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(SnapshotError, match="data hash"):
        load_snapshot(path)


def test_pinned_manifest_detects_metadata_or_joint_replacement(tmp_path):
    path = tmp_path / "bars.csv"
    manifest = write(path)
    data = json.loads(manifest_path(path).read_text())
    data["source"] = "DIFFERENT_SOURCE"
    manifest_path(path).write_text(json.dumps(data))
    with pytest.raises(SnapshotError, match="manifest hash"):
        load_snapshot(path, expected_manifest_sha256=manifest.digest)


@pytest.mark.parametrize("change", [{"row_count": 1}, {"provenance": "SYNTHETIC"}, {"symbols": ["OTHER"]}, {"start": "2026-01-05T03:40:00+00:00"}, {"schema_version": 2}])
def test_unpinned_manifest_still_checks_schema_and_record_claims(tmp_path, change):
    path = tmp_path / "bars.csv"
    write(path)
    data = json.loads(manifest_path(path).read_text())
    data.update(change)
    manifest_path(path).write_text(json.dumps(data))
    with pytest.raises(SnapshotError):
        load_snapshot(path)


@pytest.mark.parametrize("metadata", [
    {"api_key": "secret"}, {"access_token": "secret"}, {"clientcode": "private"},
    {"provider": {"password": "secret"}}, {"endpoint": "/quote?token=secret"},
    {"api_origin": "https://user:secret@example.com"},
    {"license_reference": "https://example.com?api_key=secret"},
])
def test_only_allowlisted_noncredential_source_metadata_can_be_persisted(tmp_path, metadata):
    path = tmp_path / "bars.csv"
    with pytest.raises(SnapshotError) as error:
        write(path, source_metadata=metadata)
    assert "secret" not in str(error.value)
    assert not path.exists() and not manifest_path(path).exists()


def test_synthetic_data_cannot_be_mislabelled_historical(tmp_path):
    with pytest.raises(SnapshotError, match="provenance"):
        write_snapshot(tmp_path / "bars.csv", synthetic_bars("ABC", START, [100]), source="FIXTURE", provenance="HISTORICAL", captured_at=CAPTURED)


def test_missing_sidecar_and_injected_secret_metadata_fail_closed(tmp_path):
    path = tmp_path / "bars.csv"
    write(path)
    data = json.loads(manifest_path(path).read_text())
    data["source_metadata"]["authorization"] = "secret"
    manifest_path(path).write_text(json.dumps(data))
    with pytest.raises(SnapshotError, match="manifest"):
        load_snapshot(path)
    manifest_path(path).unlink()
    with pytest.raises(SnapshotError, match="unavailable"):
        load_snapshot(path)


def test_failed_publication_removes_only_own_artifacts_and_staging_files(tmp_path):
    import os

    path = tmp_path / "bars.csv"
    link = os.link
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("simulated disk failure")
        return link(source, destination)

    with patch("trading_agent.data.persistence.os.link", side_effect=fail_second):
        with pytest.raises(OSError):
            write(path)
    assert list(tmp_path.iterdir()) == []


def test_missing_parquet_dependency_is_explicit_and_leaves_no_snapshot(tmp_path):
    path = tmp_path / "bars.parquet"
    with patch("trading_agent.data.persistence._parquet_modules", side_effect=SnapshotDependencyError("Parquet snapshots require pyarrow")):
        with pytest.raises(SnapshotDependencyError, match="pyarrow"):
            write(path)
    assert list(tmp_path.iterdir()) == []
