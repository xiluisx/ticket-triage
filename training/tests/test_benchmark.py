from training.benchmark import metrics_row, to_markdown
from triage_core.classify import Prediction


def preds(labels, conf=0.9, cost=0.0):
    return [
        Prediction(lab, {lab: conf, "x": 1 - conf}, cost_usd=cost, latency_ms=10.0)
        for lab in labels
    ]


def test_metrics_row_priority_includes_urgent_recall():
    truth = ["urgent", "urgent", "not_urgent", "not_urgent"]
    row = metrics_row(
        "classic",
        "priority",
        truth,
        preds(["urgent", "not_urgent", "not_urgent", "not_urgent"]),
        ["urgent", "not_urgent"],
        n_boot=50,
    )
    assert row["urgent_recall"] == 0.5
    assert row["accuracy"] == 0.75
    assert row["n"] == 4
    assert len(row["macro_f1_ci"]) == 2


def test_metrics_row_cost_per_1k():
    row = metrics_row(
        "claude", "category", ["a", "b"], preds(["a", "b"], cost=0.0002), ["a", "b"], n_boot=50
    )
    assert row["usd_per_1k"] == 0.2
    assert "urgent_recall" not in row


def test_to_markdown_has_header_and_rows():
    row = metrics_row("jev", "category", ["a", "b"], preds(["a", "b"]), ["a", "b"], n_boot=50)
    md = to_markdown({"category": {"test": [row]}})
    assert "| Modelo |" in md and "| jev |" in md
