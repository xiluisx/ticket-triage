import json

import pytest
from triage_core.labels import Label, LabelSet


def make_set() -> LabelSet:
    return LabelSet(
        (
            Label("cards", "Credit cards", "Card charges and fees"),
            Label("mortgage", "Mortgages", "Home loans"),
        )
    )


def test_ids_preserve_order():
    assert make_set().ids == ("cards", "mortgage")


def test_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        LabelSet((Label("a", "A", ""), Label("a", "A2", "")))


def test_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        LabelSet(())


def test_hash_is_stable_and_changes_with_content():
    assert make_set().labels_hash() == make_set().labels_hash()
    changed = LabelSet(
        (
            Label("cards", "Credit cards", "CHANGED"),
            Label("mortgage", "Mortgages", "Home loans"),
        )
    )
    assert changed.labels_hash() != make_set().labels_hash()


def test_from_json_ignores_extra_keys(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "labels": [
                    {"id": "cards", "name": "Credit cards", "description": "Card charges and fees"}
                ],
                "cfpb_product_map": {"Credit card": "cards"},
            }
        )
    )
    loaded = LabelSet.from_json(path)
    assert loaded.ids == ("cards",)
    assert loaded.labels[0].description == "Card charges and fees"


def test_descriptions_dict():
    assert make_set().descriptions() == {
        "cards": "Card charges and fees",
        "mortgage": "Home loans",
    }
