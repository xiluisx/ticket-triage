import pytest

from training.jsonl import read_jsonl, write_jsonl
from training.priority import to_binary


def test_to_binary_maps_levels():
    rows = [
        {"complaint_id": "1", "priority": "P1"},
        {"complaint_id": "2", "priority": "P2"},
        {"complaint_id": "3", "priority": "P3"},
    ]
    assert [r["priority"] for r in to_binary(rows)] == ["urgent", "not_urgent", "not_urgent"]


def test_to_binary_keeps_other_fields_and_input():
    rows = [{"complaint_id": "1", "priority": "P1", "category": "cards"}]
    out = to_binary(rows)
    assert out == [{"complaint_id": "1", "priority": "urgent", "category": "cards"}]
    assert rows[0]["priority"] == "P1"


def test_to_binary_rejects_unknown():
    with pytest.raises(ValueError, match="unknown priority"):
        to_binary([{"complaint_id": "9", "priority": "P4"}])


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "sub" / "x.jsonl"
    rows = [{"a": 1, "t": "ñ"}, {"a": 2, "t": "b"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
