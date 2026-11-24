"""Evaluation frames saved as parquet: complaint_id, text, truth, tags, product_category.

category: val (500, product labels), test (1,000, product labels), golden (100, human)
priority: val (~400, Claude labels), golden (100, human)
"""

from __future__ import annotations

import pandas as pd

from training.jsonl import read_jsonl
from training.paths import DISTILLED, EVAL, GOLDEN_V2
from training.splits import load_meta, read_ids, with_text

EVAL_SETS = {"category": ("val", "test", "golden"), "priority": ("val", "golden")}
COLUMNS = ["complaint_id", "text", "truth", "tags", "product_category"]


def golden_frame(rows: list[dict], meta: pd.DataFrame, task: str) -> pd.DataFrame:
    ids = [r["complaint_id"] for r in rows]
    return pd.DataFrame(
        {
            "complaint_id": ids,
            "text": [r["text"] for r in rows],
            "truth": [r[task] for r in rows],
            "tags": meta.loc[ids, "tags"].to_numpy(),
            "product_category": meta.loc[ids, "category"].to_numpy(),
        }
    )


def split_frame(meta: pd.DataFrame, ids: list[str], truth: list[str] | None) -> pd.DataFrame:
    frame = with_text(meta, ids)
    frame["product_category"] = frame["category"]
    frame["truth"] = frame["category"] if truth is None else truth
    return frame[COLUMNS]


def load_eval(task: str, set_name: str) -> pd.DataFrame:
    return pd.read_parquet(EVAL / f"{task}_{set_name}.parquet")


def main() -> None:
    meta = load_meta()
    golden = read_jsonl(GOLDEN_V2)
    distilled = [r for r in read_jsonl(DISTILLED) if r["split"] == "val"]
    frames = {
        ("category", "val"): split_frame(meta, read_ids("val_category"), None),
        ("category", "test"): split_frame(meta, read_ids("test_category"), None),
        ("category", "golden"): golden_frame(golden, meta, "category"),
        ("priority", "val"): split_frame(
            meta, [r["complaint_id"] for r in distilled], [r["label"] for r in distilled]
        ),
        ("priority", "golden"): golden_frame(golden, meta, "priority"),
    }
    EVAL.mkdir(parents=True, exist_ok=True)
    for (task, set_name), frame in frames.items():
        frame.to_parquet(EVAL / f"{task}_{set_name}.parquet", index=False)
        print(f"{task}/{set_name}: {len(frame):,}")


if __name__ == "__main__":
    main()
