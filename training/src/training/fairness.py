"""Fairness: metrics per group (tag, text length, readability) with bootstrap intervals.

Tags are only used to measure, never as features. `fairness.json` feeds the CI gate
(subproject 4): fail when gap_challenger - gap_champion > 0.02.

    python -m training.fairness
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from training.evalsets import load_eval  # noqa: E402
from training.golden import TAG_GROUPS  # noqa: E402
from training.metrics import bootstrap_ci, macro_f1, recall_of  # noqa: E402
from training.paths import FIGURES, REPORTS  # noqa: E402
from training.predict import model_preds  # noqa: E402
from training.replay import common_ids, replay_cascade  # noqa: E402
from training.tasks import labels_for  # noqa: E402
from training.train import latest_artifact  # noqa: E402

VOWEL_GROUPS = re.compile(r"[aeiouy]+")
WORDS = re.compile(r"[A-Za-z]+")
SENTENCE_ENDS = re.compile(r"[.!?]+")
MIN_N = 5


def syllables(word: str) -> int:
    w = word.lower()
    n = len(VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith("le") and n > 1:
        n -= 1
    return max(n, 1)


def flesch(text: str) -> float:
    words = WORDS.findall(text)
    if not words:
        return 0.0
    sentences = max(1, len(SENTENCE_ENDS.findall(text)))
    syl = sum(syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syl / len(words))


def quartile_labels(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)


def group_masks(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    df = df.reset_index(drop=True)
    tags = df["tags"].fillna("")
    tag_masks = {tag: tags.str.contains(tag) for tag in TAG_GROUPS}
    tag_masks["No tag"] = tags == ""
    length = quartile_labels(df["text"].str.split().str.len())
    readability = quartile_labels(df["text"].map(flesch))
    return {
        "tag": tag_masks,
        "length": {q: length == q for q in ("Q1", "Q2", "Q3", "Q4")},
        "readability": {q: readability == q for q in ("Q1", "Q2", "Q3", "Q4")},
    }


def per_group(
    truth, pred, masks: dict[str, pd.Series], metric, min_n: int = MIN_N, n_boot: int = 1000
) -> dict:
    truth, pred = np.asarray(truth), np.asarray(pred)
    out = {}
    for name, mask in masks.items():
        m = np.asarray(mask)
        n = int(m.sum())
        if n < min_n:
            out[name] = {"value": None, "lo": None, "hi": None, "n": n}
            continue
        lo, hi = bootstrap_ci(truth[m], pred[m], metric, n_boot=n_boot)
        out[name] = {"value": metric(truth[m], pred[m]), "lo": lo, "hi": hi, "n": n}
    return out


def gap(groups: dict) -> float | None:
    values = [g["value"] for g in groups.values()]
    if any(v is None for v in values) or len(values) < 2:
        return None
    return float(max(values) - min(values))


def predictions_with_cascade(task: str, set_name: str):
    frame = load_eval(task, set_name)
    preds = model_preds(task, set_name)
    keep = common_ids(frame["complaint_id"].tolist(), preds)
    labels_by_model = {m: [p[c].label for c in keep] for m, p in preds.items()}
    levels = None
    config_path = latest_artifact(task) / "thresholds.json"
    if config_path.exists() and "claude" in preds:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cascade = replay_cascade(config, preds).predict(keep, labels_for(task))
        labels_by_model["cascade"] = [p.label for p in cascade]
        levels = [p.level for p in cascade]
    sub = frame.set_index("complaint_id").loc[keep].reset_index()
    return sub, labels_by_model, levels


def evaluate(task: str, set_name: str, metric, metric_name: str) -> dict:
    sub, labels_by_model, levels = predictions_with_cascade(task, set_name)
    masks = group_masks(sub)
    models = {}
    for model, pred in labels_by_model.items():
        models[model] = {}
        for dim, dim_masks in masks.items():
            groups = per_group(sub["truth"], pred, dim_masks, metric)
            models[model][dim] = {"groups": groups, "gap": gap(groups)}
    result = {"metric": metric_name, "eval_set": set_name, "n": len(sub), "models": models}
    if levels is not None:
        lv = np.asarray(levels)
        result["cascade_levels"] = {
            dim: {
                g: {
                    lvl: float(np.mean(lv[np.asarray(m)] == lvl)) if m.sum() else None
                    for lvl in sorted(set(levels))
                }
                for g, m in dim_masks.items()
            }
            for dim, dim_masks in masks.items()
        }
    return result


def teacher_bias() -> dict:
    """Urgent rate of Claude vs human on the golden set, per tag group (distillation bias)."""
    sub, labels_by_model, _ = predictions_with_cascade("priority", "golden")
    human = sub["truth"].to_numpy() == "urgent"
    claude = np.asarray(labels_by_model["claude"]) == "urgent"
    out = {}
    for name, mask in group_masks(sub)["tag"].items():
        m = np.asarray(mask)
        out[name] = {
            "n": int(m.sum()),
            "human_urgent_rate": float(human[m].mean()) if m.sum() else None,
            "claude_urgent_rate": float(claude[m].mean()) if m.sum() else None,
        }
    return out


def to_markdown(report: dict) -> str:
    lines = ["# Equidad", ""]
    for key, block in report.items():
        if key == "teacher_bias":
            lines += [
                "## Sesgo del maestro (tasa de urgent, golden)",
                "",
                "| Grupo | n | Humano | Claude |",
                "|---|---|---|---|",
            ]
            for g, v in block.items():
                h = "—" if v["human_urgent_rate"] is None else f"{v['human_urgent_rate']:.2f}"
                c = "—" if v["claude_urgent_rate"] is None else f"{v['claude_urgent_rate']:.2f}"
                lines.append(f"| {g} | {v['n']} | {h} | {c} |")
            lines.append("")
            continue
        lines += [f"## {key} · {block['metric']} · {block['eval_set']} (n={block['n']})", ""]
        for model, dims in block["models"].items():
            for dim, data in dims.items():
                g = "—" if data["gap"] is None else f"{data['gap']:.3f}"
                lines += [
                    f"**{model} · {dim}** (brecha {g})",
                    "",
                    "| Grupo | n | Valor [IC 95%] |",
                    "|---|---|---|",
                ]
                for name, v in data["groups"].items():
                    val = (
                        "n insuficiente"
                        if v["value"] is None
                        else (f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
                    )
                    lines.append(f"| {name} | {v['n']} | {val} |")
                lines.append("")
    return "\n".join(lines)


def plot_category(block: dict) -> None:
    models = [m for m in ("classic", "cascade") if m in block["models"]]
    dims = ("tag", "length", "readability")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, dim in zip(axes, dims, strict=True):
        groups = list(block["models"][models[0]][dim]["groups"])
        x = np.arange(len(groups))
        width = 0.8 / len(models)
        for k, model in enumerate(models):
            data = block["models"][model][dim]["groups"]
            vals = [data[g]["value"] or 0.0 for g in groups]
            ax.bar(x + k * width, vals, width, label=model)
        ax.set_xticks(x + width * (len(models) - 1) / 2, groups, rotation=20)
        ax.set_title(dim)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (test)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fairness_category.png", dpi=150)


def main() -> None:
    category_ids = list(labels_for("category").ids)

    def f1(yt, yp):
        return macro_f1(yt, yp, category_ids)

    report = {
        "category": evaluate("category", "test", f1, "macro_f1"),
        "priority_golden": evaluate("priority", "golden", recall_of("urgent"), "urgent_recall"),
        "priority_val": evaluate("priority", "val", recall_of("urgent"), "urgent_recall"),
        "teacher_bias": teacher_bias(),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "fairness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS / "fairness.md").write_text(to_markdown(report), encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_category(report["category"])
    print(f"→ {REPORTS / 'fairness.md'}")


if __name__ == "__main__":
    main()
