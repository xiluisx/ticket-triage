import json

from training.jsonl import write_json


def test_write_json_ends_with_newline_and_creates_parent(tmp_path):
    path = tmp_path / "sub" / "x.json"
    write_json(path, {"a": 1})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == {"a": 1}
