import numpy as np

from training.fairness_gate import (
    build_report,
    gap,
    group_value,
    paired_delta,
    per_group,
    tag_masks,
)
from training.metrics import recall_of

URGENT = recall_of("urgent")


def golden(n_per_group=30):
    tags = ["Older American"] * n_per_group + [""] * n_per_group + ["Servicemember"] * n_per_group
    truth = ["urgent", "not_urgent"] * (3 * n_per_group // 2)
    return tags, truth


def test_tag_masks_handles_missing_and_combined_tags():
    masks = tag_masks(["Older American, Servicemember", None, ""])
    assert masks["Older American"].tolist() == [True, False, False]
    assert masks["Servicemember"].tolist() == [True, False, False]
    assert masks["No tag"].tolist() == [False, True, True]


def test_group_value_is_none_below_min_positives():
    truth = np.array(["urgent"] * 3 + ["not_urgent"] * 10)
    value, n, n_pos = group_value(truth, truth, URGENT, min_n=5, positive="urgent", min_positives=5)
    assert value is None and n == 13 and n_pos == 3


def test_gap_ignores_none_groups():
    groups = {"a": {"value": 0.9}, "b": {"value": None}, "c": {"value": 0.6}}
    assert abs(gap(groups) - 0.3) < 1e-9
    assert gap({"a": {"value": 0.9}, "b": {"value": None}}) is None


def test_per_group_reports_n_pos_and_none():
    truth = ["urgent"] * 2 + ["not_urgent"] * 8 + ["urgent"] * 6 + ["not_urgent"] * 4
    masks = {"g1": np.array([True] * 10 + [False] * 10), "g2": np.array([False] * 10 + [True] * 10)}
    out = per_group(truth, truth, masks, URGENT, positive="urgent", n_boot=50)
    assert out["g1"]["value"] is None and out["g1"]["n_pos"] == 2
    assert out["g2"]["value"] == 1.0 and out["g2"]["n_pos"] == 6


def test_paired_delta_passes_when_models_identical():
    tags, truth = golden()
    masks = tag_masks(tags)
    result = paired_delta(
        np.array(truth), np.array(truth), np.array(truth), masks, URGENT, "urgent", n_boot=200
    )
    assert result["status"] == "passed"
    assert result["delta_gap"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def test_paired_delta_fails_when_challenger_hurts_one_group():
    tags, truth = golden()
    challenger = [
        "not_urgent" if tag == "Older American" else label
        for tag, label in zip(tags, truth, strict=True)
    ]
    result = paired_delta(
        np.array(truth),
        np.array(truth),
        np.array(challenger),
        tag_masks(tags),
        URGENT,
        "urgent",
        n_boot=200,
    )
    assert result["status"] == "failed"
    assert result["ci95"][0] > 0


def test_build_report_without_champion_is_skipped():
    tags, truth = golden()
    report = build_report("priority", truth, tags, challenger=("v1", truth), n_boot=50)
    assert report["schema_version"] == 2
    assert report["metric"] == "recall_urgent"
    assert set(report["models"]) == {"challenger"}
    assert report["gate"]["status"] == "skipped"
    assert report["gate"]["reason"] == "no champion"
    group = report["models"]["challenger"]["per_group"]["Older American"]
    assert group["n"] == 30 and group["n_pos"] == 15 and group["value"] == 1.0
    assert len(group["ci95"]) == 2


def test_build_report_with_champion_runs_gate():
    tags, truth = golden()
    report = build_report(
        "category", truth, tags, challenger=("v2", truth), champion=("v1", truth), n_boot=50
    )
    assert report["metric"] == "macro_f1_present"
    assert report["gate"]["status"] == "passed"
    assert report["models"]["champion"]["version"] == "v1"
