import pandas as pd

from training.splits import assign_split, build_splits, read_ids, stratified_sample, write_ids


def meta(n=600):
    dates = pd.to_datetime(["2026-03-15"] * 300 + ["2026-05-10"] * 150 + ["2026-07-01"] * 150)
    df = pd.DataFrame(
        {
            "complaint_id": [str(i) for i in range(n)],
            "date_received": dates[:n],
            "category": [["a", "b", "c"][i % 3] for i in range(n)],
            "tags": ["" for _ in range(n)],
        }
    )
    df["split"] = assign_split(df["date_received"])
    return df.set_index("complaint_id", drop=False)


def test_assign_split_boundaries():
    dates = pd.Series(pd.to_datetime(["2026-04-30", "2026-05-01", "2026-05-31", "2026-06-01"]))
    assert assign_split(dates).tolist() == ["train", "val", "val", "test"]


def test_stratified_sample_size_and_proportions():
    out = stratified_sample(meta(), 90, "category")
    assert len(out) == 90
    assert out["category"].value_counts().to_dict() == {"a": 30, "b": 30, "c": 30}


def test_stratified_sample_returns_all_when_n_large():
    assert len(stratified_sample(meta(), 10_000, "category")) == 600


def test_build_splits_excludes_golden_and_respects_periods():
    m = meta()
    golden = {"500", "501"}
    sizes = {"train_category": 60, "val_category": 30, "test_category": 30}
    splits = build_splits(m, golden, sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert not golden & set(splits["test_category"])
    assert set(m.loc[splits["val_category"], "split"]) == {"val"}
    assert set(m.loc[splits["test_category"], "split"]) == {"test"}


def test_build_splits_is_deterministic():
    sizes = {"train_category": 60, "val_category": 30, "test_category": 30}
    assert build_splits(meta(), set(), sizes) == build_splits(meta(), set(), sizes)


def test_ids_roundtrip(tmp_path):
    write_ids("x", ["3", "1"], root=tmp_path)
    assert read_ids("x", root=tmp_path) == ["3", "1"]
