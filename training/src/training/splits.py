"""Temporal splits over the CFPB parquet: train <= 2026-04, val 2026-05, test 2026-06..08.

Same cut as the baseline (train < 2026-06), with May carved out for calibration and thresholds.
Only complaint ids are stored; narratives are read on demand (memory-bounded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from training.cfpb import load_product_map, map_categories, read_narratives
from training.jsonl import read_jsonl
from training.paths import CATEGORIES, CFPB, GOLDEN_V2, SPLITS

VAL_START = pd.Timestamp("2026-05-01")
TEST_START = pd.Timestamp("2026-06-01")
SIZES = {"train_category": 50_000, "val_category": 500, "test_category": 1_000}
SEED = 42


def assign_split(dates: pd.Series) -> pd.Series:
    values = np.select([dates < VAL_START, dates < TEST_START], ["train", "val"], "test")
    return pd.Series(values, index=dates.index)


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int = SEED) -> pd.DataFrame:
    if n >= len(df):
        return df
    picked, _ = train_test_split(df, train_size=n, stratify=df[by], random_state=seed)
    return picked


def load_meta(path: Path = CFPB) -> pd.DataFrame:
    """Small columns of every mapped complaint, indexed by complaint_id; `row` = parquet row."""
    meta = pq.read_table(
        path, columns=["complaint_id", "date_received", "product", "tags"]
    ).to_pandas()
    meta["row"] = np.arange(len(meta))
    meta["tags"] = meta["tags"].fillna("")
    meta["category"] = map_categories(meta["product"], load_product_map(CATEGORIES))
    meta = meta.dropna(subset=["category"])
    meta["split"] = assign_split(meta["date_received"])
    return meta.set_index("complaint_id", drop=False)


def build_splits(
    meta: pd.DataFrame, golden_ids: set[str], sizes: dict[str, int] = SIZES, seed: int = SEED
) -> dict[str, list[str]]:
    pools = {
        "train_category": meta[meta["split"] == "train"],
        "val_category": meta[meta["split"] == "val"],
        "test_category": meta[(meta["split"] == "test") & ~meta["complaint_id"].isin(golden_ids)],
    }
    return {
        name: sorted(stratified_sample(pools[name], n, "category", seed)["complaint_id"], key=int)
        for name, n in sizes.items()
    }


def write_ids(name: str, ids: list[str], root: Path = SPLITS) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_ids(name: str, root: Path = SPLITS) -> list[str]:
    return (root / f"{name}.txt").read_text(encoding="utf-8").split()


def with_text(meta: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    sub = meta.loc[ids].copy()
    sub["text"] = read_narratives(CFPB, sub["row"].to_numpy())
    return sub.reset_index(drop=True)


if __name__ == "__main__":
    meta = load_meta()
    golden_ids = {r["complaint_id"] for r in read_jsonl(GOLDEN_V2)}
    for name, ids in build_splits(meta, golden_ids).items():
        write_ids(name, ids)
        print(f"{name}: {len(ids):,}")
