import pandas as pd

from training.distill import kappa_report, sample_for_distill


def meta():
    rows = []
    for split, n in (("train", 2000), ("val", 600)):
        for i in range(n):
            tag = "Older American" if i % 10 == 0 else "Servicemember" if i % 10 == 1 else ""
            rows.append(
                {
                    "complaint_id": f"{split}{i}",
                    "split": split,
                    "tags": tag,
                    "category": ["a", "b", "c", "d"][i % 4],
                }
            )
    return pd.DataFrame(rows).set_index("complaint_id", drop=False)


def test_sample_sizes_per_split():
    out = sample_for_distill(meta(), sizes={"train": 400, "val": 100}, per_tag_total=100)
    assert out["split"].value_counts().to_dict() == {"train": 400, "val": 100}
    assert out["complaint_id"].is_unique


def test_sample_oversamples_tags():
    out = sample_for_distill(meta(), sizes={"train": 400, "val": 100}, per_tag_total=100)
    # 10% of the pool has each tag; quotas force ~20% in the sample
    assert out["tags"].str.contains("Older American").sum() >= 100
    assert out["tags"].str.contains("Servicemember").sum() >= 100


def test_sample_is_deterministic():
    a = sample_for_distill(meta(), sizes={"train": 400, "val": 100}, per_tag_total=100)
    b = sample_for_distill(meta(), sizes={"train": 400, "val": 100}, per_tag_total=100)
    assert a["complaint_id"].tolist() == b["complaint_id"].tolist()


def test_kappa_report():
    report = kappa_report(
        ["urgent", "not_urgent", "urgent"], ["urgent", "not_urgent", "not_urgent"]
    )
    assert report["n"] == 3
    assert 0 < report["kappa"] < 1
    assert report["confusion"]["urgent"]["not_urgent"] == 1
