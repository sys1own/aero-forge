"""Input validation, JSON parsing, and result serialization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from batch_analytics import Record
from batch_analytics import AggregateResult


def parse_records(data: Any) -> List[Record]:
    """Convert a JSON list of record dicts into `Record` objects."""
    if not isinstance(data, list):
        raise ValueError("records must be a JSON list")

    records: List[Record] = []
    for idx, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise ValueError(f"record at index {idx} must be an object")
        try:
            record_id = str(item["id"])
            value = float(item["value"])
            timestamp = float(item.get("timestamp", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid record at index {idx}: {exc}") from exc
        records.append(Record(record_id, value, timestamp))
    return records


def serialize_record(record: Record) -> Dict[str, Any]:
    """Serialize a `Record` to a plain dict."""
    return {
        "id": record.id,
        "value": record.value,
        "timestamp": record.timestamp,
    }


def serialize_aggregate(result: AggregateResult) -> Dict[str, Any]:
    """Serialize an `AggregateResult` to a plain dict."""
    return {
        "window_start": result.window_start,
        "mean": result.mean,
        "std": result.std,
        "outliers": list(result.outliers),
    }
