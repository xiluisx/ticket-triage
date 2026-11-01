import json
from pathlib import Path

from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[2]


def test_categories_file_is_valid():
    path = ROOT / "datasets" / "categories_v1.json"
    labels = LabelSet.from_json(path)
    assert len(labels.ids) == 8
    product_map = json.loads(path.read_text())["cfpb_product_map"]
    assert set(product_map.values()) == set(labels.ids)


def test_priorities_file_is_valid():
    labels = LabelSet.from_json(ROOT / "datasets" / "priorities_v1.json")
    assert labels.ids == ("P1", "P2", "P3")
