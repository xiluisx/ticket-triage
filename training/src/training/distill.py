"""Priority distillation: Claude labels ~2k complaints that train the classic priority model.

python -m training.distill gate   # Claude vs human on the golden set (Cohen's kappa)
python -m training.distill run    # label the 2k sample
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from training.cache import cache_for, confirm_cost, predict_cached
from training.golden import TAG_GROUPS
from training.jsonl import read_jsonl, write_jsonl
from training.llm import claude_classifier, estimate_claude_usd
from training.metrics import kappa
from training.paths import DISTILL, DISTILLED, GOLDEN_V2, REPORTS
from training.splits import load_meta, with_text
from training.tasks import labels_for

SIZES = {"train": 1600, "val": 400}
PER_TAG_TOTAL = 200
KAPPA_MIN = 0.6
SEED = 42


def sample_for_distill(
    meta: pd.DataFrame,
    sizes: dict[str, int] = SIZES,
    per_tag_total: int = PER_TAG_TOTAL,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per split: tag quotas first (oversampling fairness groups), then stratified fill."""
    total = sum(sizes.values())
    parts = []
    for split, size in sizes.items():
        pool = meta[meta["split"] == split]
        picked: list[str] = []
        for tag in TAG_GROUPS:
            quota = round(per_tag_total * size / total)
            candidates = pool[pool["tags"].str.contains(tag) & ~pool.index.isin(picked)]
            sampled = candidates.sample(min(quota, len(candidates)), random_state=seed)
            picked += sampled.index.tolist()
        rest = pool[~pool.index.isin(picked)]
        fill, _ = train_test_split(
            rest, train_size=size - len(picked), stratify=rest["category"], random_state=seed
        )
        parts.append(pd.concat([pool.loc[picked], fill]).assign(split=split))
    return pd.concat(parts).reset_index(drop=True)


def kappa_report(human: list[str], model: list[str]) -> dict:
    confusion: dict[str, dict[str, int]] = {}
    for h, m in zip(human, model, strict=True):
        confusion.setdefault(h, {}).setdefault(m, 0)
        confusion[h][m] += 1
    return {"n": len(human), "kappa": kappa(human, model), "confusion": confusion}


def gate(assume_yes: bool) -> None:
    rows = read_jsonl(GOLDEN_V2)
    clf = claude_classifier("priority")
    cache = cache_for(clf)
    ids = [r["complaint_id"] for r in rows]
    texts = [r["text"] for r in rows]
    missing = [t for cid, t in zip(ids, texts, strict=True) if cid not in cache]
    confirm_cost(len(missing), estimate_claude_usd(missing, clf.model_id), assume_yes)
    preds = predict_cached(
        clf, cache, ids, texts, labels_for("priority"), failed_path=DISTILL / "gate_failed.jsonl"
    )
    done = [r for r in rows if r["complaint_id"] in preds]
    report = kappa_report(
        [r["priority"] for r in done], [preds[r["complaint_id"]].label for r in done]
    )
    mean_cost = sum(p.cost_usd for p in preds.values()) / len(preds)
    report |= {
        "model_id": clf.model_id,
        "cache_key": clf.cache_key,
        "passed": report["kappa"] >= KAPPA_MIN,
        "mean_usd_per_call": mean_cost,
    }
    write_jsonl(
        DISTILL / "gate_disagreements.jsonl",
        [
            {
                "complaint_id": r["complaint_id"],
                "human": r["priority"],
                "claude": preds[r["complaint_id"]].label,
                "text": r["text"][:600],
            }
            for r in done
            if preds[r["complaint_id"]].label != r["priority"]
        ],
    )
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "kappa_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"kappa={report['kappa']:.3f} (n={report['n']}, min {KAPPA_MIN}) → {verdict}")
    print(f"confusion (human → claude): {report['confusion']}")
    print(f"mean cost/call ${mean_cost:.6f}; projected 2k run ≈ ${mean_cost * 2000:.2f}")
    print(f"disagreements → {DISTILL / 'gate_disagreements.jsonl'}")


def run(assume_yes: bool, workers: int) -> None:
    meta = load_meta()
    sample = sample_for_distill(meta[meta["split"].isin(list(SIZES))])
    frame = with_text(meta, sample["complaint_id"].tolist()).assign(
        split=sample["split"].to_numpy()
    )
    clf = claude_classifier("priority")
    cache = cache_for(clf)
    ids = frame["complaint_id"].tolist()
    texts = frame["text"].tolist()
    missing = [t for cid, t in zip(ids, texts, strict=True) if cid not in cache]
    confirm_cost(len(missing), estimate_claude_usd(missing, clf.model_id), assume_yes)
    preds = predict_cached(
        clf,
        cache,
        ids,
        texts,
        labels_for("priority"),
        workers=workers,
        failed_path=DISTILL / "distill_failed.jsonl",
    )
    rows = [
        {
            "complaint_id": cid,
            "split": split,
            "label": preds[cid].label,
            "model_id": clf.model_id,
            "cache_key": clf.cache_key,
        }
        for cid, split in zip(ids, frame["split"], strict=True)
        if cid in preds
    ]
    write_jsonl(DISTILLED, rows)
    print(f"{len(rows)} labels → {DISTILLED}")
    print(Counter((r["split"], r["label"]) for r in rows))
    print(f"cost ${sum(p.cost_usd for p in preds.values()):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["gate", "run"])
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "gate":
        gate(args.yes)
    else:
        run(args.yes, args.workers)
