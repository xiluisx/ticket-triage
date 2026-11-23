import pandas as pd
import pytest

from training.golden import (
    export_annotations,
    label_studio_config,
    merge_annotations,
    sample_golden,
    subsample_golden,
)
from triage_core.labels import Label, LabelSet

CATS = LabelSet((Label("a", "A", "cat a"), Label("b", "B", "cat b"), Label("c", "C", "cat c")))
PRIOS = LabelSet(
    (Label("P1", "Urgent", "now"), Label("P2", "Standard", "soon"), Label("P3", "Low", "later"))
)


def frame(n: int = 600) -> pd.DataFrame:
    tags = [
        "Older American" if i % 20 == 0 else "Servicemember" if i % 20 == 1 else ""
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            "complaint_id": [str(i) for i in range(n)],
            "narrative": [f"text {i}" for i in range(n)],
            "product": ["P"] * n,
            "category": [["a", "b", "c"][i % 3] if i % 50 else "c" for i in range(n)],
            "tags": tags,
        }
    )


def test_sample_golden_respects_quotas_and_size():
    out = sample_golden(frame(), n=60, min_per_category=10, min_per_tag=8, seed=1)
    assert len(out) == 60
    assert out["complaint_id"].is_unique
    assert out["category"].value_counts().min() >= 10
    assert out["tags"].str.contains("Older American").sum() >= 8
    assert out["tags"].str.contains("Servicemember").sum() >= 8


def test_sample_golden_is_deterministic():
    a = sample_golden(frame(), n=60, min_per_category=10, min_per_tag=8, seed=1)
    b = sample_golden(frame(), n=60, min_per_category=10, min_per_tag=8, seed=1)
    assert a["complaint_id"].tolist() == b["complaint_id"].tolist()


def test_sample_golden_rejects_quotas_larger_than_n():
    with pytest.raises(ValueError, match="Quotas"):
        sample_golden(frame(), n=10, min_per_category=10, min_per_tag=8, seed=1)


def test_label_studio_config_lists_all_choices():
    xml = label_studio_config(CATS, PRIOS)
    for value in ["a", "b", "c", "P1", "P2", "P3"]:
        assert f'value="{value}"' in xml
    assert 'name="category"' in xml and 'name="priority"' in xml


def test_label_studio_config_escapes_descriptions():
    cats = LabelSet((Label("a", "A & B", 'say "hi" <now>'),))
    xml = label_studio_config(cats, PRIOS)
    assert "A &amp; B: say &quot;hi&quot; &lt;now&gt;" in xml


def ls_task(cid: str, cat: str | None, prio: str | None, cancelled: bool = False) -> dict:
    result = []
    if cat:
        result.append({"from_name": "category", "value": {"choices": [cat]}})
    if prio:
        result.append({"from_name": "priority", "value": {"choices": [prio]}})
    return {
        "data": {"complaint_id": cid, "text": f"t{cid}", "product": "P"},
        "annotations": [{"result": result, "was_cancelled": cancelled}],
    }


def test_export_annotations_converts_and_sorts():
    rows = export_annotations([ls_task("2", "b", "P2"), ls_task("1", "a", "P1")], CATS, PRIOS)
    assert rows == [
        {
            "complaint_id": "1",
            "text": "t1",
            "product_original": "P",
            "category": "a",
            "priority": "P1",
        },
        {
            "complaint_id": "2",
            "text": "t2",
            "product_original": "P",
            "category": "b",
            "priority": "P2",
        },
    ]


def test_export_skips_unfinished_and_cancelled():
    tasks = [
        ls_task("1", "a", None),
        ls_task("2", "a", "P1", cancelled=True),
        {"data": {"complaint_id": "3"}, "annotations": []},
    ]
    assert export_annotations(tasks, CATS, PRIOS) == []


def test_export_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown"):
        export_annotations([ls_task("1", "zzz", "P1")], CATS, PRIOS)


QUOTAS = {"a": 10, "b": 10, "c": 15}


def test_subsample_keeps_labeled_and_meets_quotas():
    df = frame()
    keep = {"2", "4", "5", "7"}
    out = subsample_golden(df, keep, n=60, category_quotas=QUOTAS, min_per_tag=8, seed=1)
    assert len(out) == 60
    assert out["complaint_id"].is_unique
    assert keep <= set(out["complaint_id"])
    counts = out["category"].value_counts()
    assert all(counts[c] >= q for c, q in QUOTAS.items())
    assert out["tags"].str.contains("Older American").sum() >= 8
    assert out["tags"].str.contains("Servicemember").sum() >= 8


def test_subsample_counts_labeled_toward_quotas():
    df = frame()
    keep = set(df.loc[df["category"] == "c", "complaint_id"][:15])
    out = subsample_golden(df, keep, n=60, category_quotas=QUOTAS, min_per_tag=8, seed=1)
    assert keep <= set(out["complaint_id"])
    assert len(out) == 60


def test_subsample_is_deterministic():
    args = dict(n=60, category_quotas=QUOTAS, min_per_tag=8, seed=1)
    a = subsample_golden(frame(), {"3"}, **args)
    b = subsample_golden(frame(), {"3"}, **args)
    assert a["complaint_id"].tolist() == b["complaint_id"].tolist()


def test_subsample_rejects_more_labeled_than_n():
    with pytest.raises(ValueError, match="labeled"):
        subsample_golden(frame(), {str(i) for i in range(61)}, 60, QUOTAS, 8, 1)


def test_subsample_rejects_unknown_labeled_id():
    with pytest.raises(ValueError, match="not in pool"):
        subsample_golden(frame(), {"nope"}, 60, QUOTAS, 8, 1)


def test_subsample_rejects_quotas_larger_than_n():
    with pytest.raises(ValueError, match="Quotas"):
        subsample_golden(frame(), set(), 30, QUOTAS, 8, 1)


def test_merge_annotations_attaches_results_to_selected_tasks():
    pool = [ls_task(cid, None, None) for cid in ["1", "2", "3"]]
    for t in pool:
        del t["annotations"]
    exported = [ls_task("2", "b", "P2"), ls_task("9", "a", "P1")]
    out = merge_annotations(pool, exported, ["3", "2"])
    assert [t["data"]["complaint_id"] for t in out] == ["2", "3"]
    assert out[0]["annotations"] == [{"result": exported[0]["annotations"][0]["result"]}]
    assert "annotations" not in out[1]


def test_merge_annotations_ignores_cancelled():
    pool = [{"data": {"complaint_id": "1", "text": "t", "product": "P"}}]
    out = merge_annotations(pool, [ls_task("1", "a", "P1", cancelled=True)], ["1"])
    assert "annotations" not in out[0]
