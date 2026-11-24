"""Benchmark: classic, Jev, Claude and the cascade on every evaluation set (no API calls).

python -m training.benchmark
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from training.evalsets import EVAL_SETS, load_eval  # noqa: E402
from training.metrics import (  # noqa: E402
    accuracy,
    bootstrap_ci,
    ece,
    macro_f1,
    p95,
    probs_matrix,
    recall_of,
)
from training.paths import FIGURES, REPORTS  # noqa: E402
from training.predict import model_preds  # noqa: E402
from training.replay import common_ids, replay_cascade  # noqa: E402
from training.tasks import TASKS, labels_for  # noqa: E402
from training.train import latest_artifact  # noqa: E402
from triage_core.classify import Prediction  # noqa: E402


def metrics_row(
    model: str,
    task: str,
    truth: list[str],
    preds: list[Prediction],
    classes: list[str],
    n_boot: int = 1000,
) -> dict:
    pred = [p.label for p in preds]

    def f1(yt, yp):
        return macro_f1(yt, yp, classes)

    row = {
        "model": model,
        "n": len(truth),
        "macro_f1": f1(truth, pred),
        "macro_f1_ci": list(bootstrap_ci(truth, pred, f1, n_boot=n_boot)),
        "accuracy": accuracy(truth, pred),
        "ece": ece(truth, probs_matrix(preds, classes), classes),
        "p95_ms": p95([p.latency_ms for p in preds]),
        "usd_per_1k": round(float(np.mean([p.cost_usd for p in preds])) * 1000, 6),
    }
    if task == "priority":
        urgent = recall_of("urgent")
        row["urgent_recall"] = urgent(truth, pred)
        row["urgent_recall_ci"] = list(bootstrap_ci(truth, pred, urgent, n_boot=n_boot))
    return row


def evaluate(task: str, set_name: str) -> list[dict]:
    classes = list(labels_for(task).ids)
    frame = load_eval(task, set_name)
    preds = model_preds(task, set_name)
    keep = common_ids(frame["complaint_id"].tolist(), preds)
    truth = frame.set_index("complaint_id").loc[keep, "truth"].tolist()
    rows = [metrics_row(m, task, truth, [p[c] for c in keep], classes) for m, p in preds.items()]
    config_path = latest_artifact(task) / "thresholds.json"
    if config_path.exists() and "claude" in preds:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cascade_preds = replay_cascade(config, preds).predict(keep, labels_for(task))
        row = metrics_row("cascade", task, truth, cascade_preds, classes)
        row["levels"] = {
            lvl: sum(p.level == lvl for p in cascade_preds) / len(cascade_preds)
            for lvl in config["order"]
        }
        rows.append(row)
    return rows


def fmt_ci(value: float, ci: list[float]) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def to_markdown(results: dict) -> str:
    lines = ["# Benchmark", ""]
    for task, sets in results.items():
        for set_name, rows in sets.items():
            urgent = task == "priority"
            lines += [f"## {task} · {set_name} (n={rows[0]['n'] if rows else 0})", ""]
            header = "| Modelo | macro-F1 [IC 95%] |"
            header += " recall urgent [IC 95%] |" if urgent else ""
            header += " accuracy | ECE | p95 ms | USD / 1k |"
            lines += [header, "|" + "---|" * (header.count("|") - 1)]
            for r in rows:
                cells = [r["model"], fmt_ci(r["macro_f1"], r["macro_f1_ci"])]
                if urgent:
                    cells.append(fmt_ci(r["urgent_recall"], r["urgent_recall_ci"]))
                cells += [
                    f"{r['accuracy']:.3f}",
                    f"{r['ece']:.3f}",
                    f"{r['p95_ms']:.0f}",
                    f"{r['usd_per_1k']:.4f}",
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def product_vs_golden() -> float:
    frame = load_eval("category", "golden")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        frame["truth"],
        frame["product_category"],
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_xlabel("Categoría según producto CFPB")
    ax.set_ylabel("Categoría golden (humano)")
    ax.set_title("Ruido de etiquetas: producto vs golden")
    fig.tight_layout()
    fig.savefig(FIGURES / "product_vs_golden.png", dpi=150)
    return accuracy(frame["truth"].tolist(), frame["product_category"].tolist())


def plot(results: dict) -> None:
    panels = [("category", "test", "macro_f1"), ("priority", "golden", "urgent_recall")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (task, set_name, metric) in zip(axes, panels, strict=True):
        rows = results[task][set_name]
        values = [r[metric] for r in rows]
        ci = np.array([r[f"{metric}_ci"] for r in rows])
        err = np.vstack([np.array(values) - ci[:, 0], ci[:, 1] - np.array(values)])
        ax.bar([r["model"] for r in rows], values, yerr=err, color="#2a6fdb", capsize=4)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task} · {set_name} · {metric}")
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark.png", dpi=150)


def main() -> None:
    results = {task: {s: evaluate(task, s) for s in EVAL_SETS[task]} for task in TASKS}
    FIGURES.mkdir(parents=True, exist_ok=True)
    agreement = product_vs_golden()
    REPORTS.mkdir(exist_ok=True)
    payload = {"results": results, "product_vs_golden_agreement": agreement}
    (REPORTS / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = to_markdown(results)
    md += f"\nAcuerdo producto CFPB vs golden (categoría): {agreement:.2f}\n"
    (REPORTS / "benchmark.md").write_text(md, encoding="utf-8")
    plot(results)
    print(md)


if __name__ == "__main__":
    main()
