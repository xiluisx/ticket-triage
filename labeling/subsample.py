"""Shrink the 200-task golden pool to 100, keeping every task already labeled.

Reads the original import (pool) and the current Label Studio export, tops up per-category
and per-tag quotas from unlabeled tasks, and writes a new import file that carries over the
finished annotations, so a fresh Label Studio project only shows the remaining work.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from training.cfpb import load_product_map, map_categories
from training.golden import export_annotations, merge_annotations, subsample_golden
from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden"
N = 100
MIN_PER_CATEGORY = 8
CATEGORY_OVERRIDES = {"credit_reporting": 12}  # largest category in CFPB
MIN_PER_TAG = 15

if __name__ == "__main__":
    pool_tasks = json.loads((GOLDEN / "labelstudio_import.json").read_text(encoding="utf-8"))
    exported = json.loads((GOLDEN / "labelstudio_export.json").read_text(encoding="utf-8"))
    categories = LabelSet.from_json(ROOT / "datasets" / "categories_v1.json")
    priorities = LabelSet.from_json(ROOT / "datasets" / "priorities_v1.json")
    labeled = {r["complaint_id"] for r in export_annotations(exported, categories, priorities)}

    ids = [t["data"]["complaint_id"] for t in pool_tasks]
    meta = pq.read_table(
        ROOT / "data" / "cfpb" / "complaints.parquet", columns=["complaint_id", "product", "tags"]
    ).to_pandas()
    meta = meta[meta["complaint_id"].isin(ids)].reset_index(drop=True)
    meta["tags"] = meta["tags"].fillna("")
    meta["category"] = map_categories(
        meta["product"], load_product_map(ROOT / "datasets" / "categories_v1.json")
    )

    quotas = {lb.id: MIN_PER_CATEGORY for lb in categories.labels} | CATEGORY_OVERRIDES
    golden = subsample_golden(meta, labeled, N, quotas, MIN_PER_TAG, seed=42)
    tasks = merge_annotations(pool_tasks, exported, golden["complaint_id"].tolist())

    out = GOLDEN / f"labelstudio_import_{N}.json"
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"{len(tasks)} tasks → {out} ({len(labeled)} already labeled, "
        f"{len(tasks) - len(labeled)} to go)"
    )
    print(golden["category"].value_counts().to_string())
    for tag in ("Older American", "Servicemember"):
        print(f"{tag}:", golden["tags"].str.contains(tag).sum())
