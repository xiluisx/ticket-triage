"""Threshold sweep on val: cheapest cascade that meets the quality constraint.

category: macro-F1 >= Claude-only macro-F1 - 0.01
priority: recall(urgent) >= 0.95, with not_urgent never easier to accept than urgent

python -m training.thresholds category|priority
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.evalsets import load_eval  # noqa: E402
from training.jsonl import write_json  # noqa: E402
from training.metrics import ece, macro_f1, probs_matrix, recall_of  # noqa: E402
from training.paths import FIGURES, REPORTS  # noqa: E402
from training.predict import model_preds  # noqa: E402
from training.replay import common_ids  # noqa: E402
from training.tasks import labels_for  # noqa: E402
from training.train import latest_artifact  # noqa: E402
from triage_core.cascade import LevelThresholds  # noqa: E402

NEVER = 1.01  # no probability reaches it: the level always escalates
CATEGORY_GRID = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, NEVER)
PRIORITY_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
URGENT_RECALL_MIN = 0.95
F1_TOLERANCE = 0.01
JEV_ECE_MAX = 0.10


def simulate(
    level_probs: list[np.ndarray], final_pred: np.ndarray, thresholds: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(final_pred)
    pred = np.array(final_pred, copy=True)
    level = np.full(n, len(level_probs))
    pending = np.ones(n, dtype=bool)
    for i, (probs, thr) in enumerate(zip(level_probs, thresholds, strict=True)):
        top = probs.argmax(axis=1)
        accept = pending & (probs[np.arange(n), top] >= thr[top])
        pred[accept] = top[accept]
        level[accept] = i
        pending &= ~accept
    return pred, level


def sweep(
    level_probs, final_pred, y_true, costs, candidates, score: Callable, constraint: Callable
) -> list[dict]:
    cumulative = np.cumsum(costs)
    rows = []
    for combo in itertools.product(*candidates):
        pred, level = simulate(level_probs, final_pred, list(combo))
        rows.append(
            {
                "thresholds": [c.tolist() for c in combo],
                "score": score(y_true, pred),
                "ok": bool(constraint(y_true, pred)),
                "usd_per_1k": float(cumulative[level].mean() * 1000),
                "coverage": [float(np.mean(level == i)) for i in range(len(costs))],
            }
        )
    return rows


def choose(rows: list[dict]) -> dict:
    feasible = [r for r in rows if r["ok"]]
    return min(feasible, key=lambda r: (r["usd_per_1k"], -r["score"]))


def category_candidates(k: int) -> list[np.ndarray]:
    return [np.full(k, t) for t in CATEGORY_GRID]


def priority_candidates(ids: list[str]) -> list[np.ndarray]:
    u = ids.index("urgent")
    out = []
    for t_urgent, t_other in itertools.product(PRIORITY_GRID, PRIORITY_GRID):
        if t_other >= t_urgent:
            vec = np.full(len(ids), t_other)
            vec[u] = t_urgent
            out.append(vec)
    out.append(np.full(len(ids), NEVER))
    return out


def apply_temperature_matrix(probs: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.clip(probs, 1e-12, 1.0) ** (1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray) -> float:
    rows = np.arange(len(y_idx))

    def nll(t: float) -> float:
        return float(-np.mean(np.log(apply_temperature_matrix(probs, t)[rows, y_idx])))

    return float(min(np.geomspace(0.25, 4.0, 41), key=nll))


def to_level_thresholds(vec: np.ndarray, ids: list[str]) -> LevelThresholds:
    default = float(vec.min())
    return LevelThresholds(
        default, {ids[j]: float(v) for j, v in enumerate(vec) if float(v) != default}
    )


def main(task: str) -> None:
    ids = list(labels_for(task).ids)
    frame = load_eval(task, "val")
    preds = model_preds(task, "val")
    keep = common_ids(frame["complaint_id"].tolist(), preds)
    truth = frame.set_index("complaint_id").loc[keep, "truth"].tolist()
    y_idx = np.array([ids.index(t) for t in truth])
    levels = [m for m in ("classic", "jev") if m in preds]
    level_probs = [probs_matrix([preds[m][c] for c in keep], ids) for m in levels]

    temperature = 1.0
    if "jev" in levels:
        j = levels.index("jev")
        jev_ece = ece(truth, level_probs[j], ids)
        if jev_ece > JEV_ECE_MAX:
            temperature = fit_temperature(level_probs[j], y_idx)
            level_probs[j] = apply_temperature_matrix(level_probs[j], temperature)
        print(f"Jev ECE={jev_ece:.3f} → temperature={temperature:.2f}")

    claude_idx = np.array([ids.index(preds["claude"][c].label) for c in keep])
    costs = [float(np.mean([preds[m][c].cost_usd for c in keep])) for m in [*levels, "claude"]]
    label_idx = list(range(len(ids)))

    def f1(yt, yp):
        return macro_f1(yt, yp, label_idx)

    if task == "category":
        floor = f1(y_idx, claude_idx) - F1_TOLERANCE
        candidates = [category_candidates(len(ids))] * len(levels)

        def constraint(yt, yp):
            return f1(yt, yp) >= floor
    else:
        urgent = recall_of(ids.index("urgent"))
        candidates = [priority_candidates(ids)] * len(levels)

        def constraint(yt, yp):
            return urgent(yt, yp) >= URGENT_RECALL_MIN

    rows = sweep(level_probs, claude_idx, y_idx, costs, candidates, f1, constraint)
    best = choose(rows)
    config = {
        "task": task,
        "order": [*levels, "claude"],
        "levels": {
            m: to_level_thresholds(np.array(v), ids).to_dict()
            for m, v in zip(levels, best["thresholds"], strict=True)
        },
        "jev_temperature": temperature,
        "val": best,
        "val_rows": len(keep),
    }
    artifact = latest_artifact(task)
    write_json(artifact / "thresholds.json", config)
    write_json(REPORTS / f"cascade_{task}.json", {"chosen": best, "config": config, "sweep": rows})

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    ax.scatter(
        [r["usd_per_1k"] for r in bad],
        [r["score"] for r in bad],
        s=8,
        c="#bbbbbb",
        label="no cumple restricción",
    )
    ax.scatter(
        [r["usd_per_1k"] for r in ok],
        [r["score"] for r in ok],
        s=8,
        c="#2a6fdb",
        label="cumple restricción",
    )
    ax.scatter(
        [best["usd_per_1k"]], [best["score"]], s=80, c="#d1495b", marker="*", label="elegido"
    )
    ax.set_xlabel("Costo por 1,000 tickets (USD)")
    ax.set_ylabel("macro-F1 en val")
    ax.set_title(f"Cascada · {task}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"cascade_{task}.png", dpi=150)
    print(f"chosen: {best} → {artifact / 'thresholds.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
