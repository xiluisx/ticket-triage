"""Sample 200 test-period complaints for the golden set and write a Label Studio import file.

Rows are chosen from small columns; only the 200 sampled narratives are read (memory-bounded).
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from training.cfpb import load_product_map, map_categories, read_narratives
from training.golden import sample_golden

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cfpb" / "complaints.parquet"

if __name__ == "__main__":
    meta = pq.read_table(
        DATA, columns=["complaint_id", "date_received", "product", "tags"]
    ).to_pandas()
    meta = meta[meta["date_received"] >= "2026-06-01"].copy()
    meta["category"] = map_categories(
        meta["product"], load_product_map(ROOT / "datasets" / "categories_v1.json")
    )
    meta = meta.dropna(subset=["category"])
    golden = sample_golden(meta, n=200, min_per_category=15, min_per_tag=25, seed=42)

    # sample_golden resets the index, so map back to parquet row positions via complaint_id
    rows = meta.reset_index().set_index("complaint_id").loc[golden["complaint_id"], "index"]
    narratives = read_narratives(DATA, rows.to_numpy())
    tasks = [
        {"data": {"complaint_id": cid, "text": text, "product": product}}
        for cid, text, product in zip(
            golden["complaint_id"], narratives, golden["product"], strict=True
        )
    ]
    out = ROOT / "data" / "golden" / "labelstudio_import.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks → {out}")
    print(golden["category"].value_counts().to_string())
    print("Older American:", golden["tags"].str.contains("Older American").sum())
    print("Servicemember:", golden["tags"].str.contains("Servicemember").sum())
