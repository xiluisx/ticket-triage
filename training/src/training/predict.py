"""Run classic, Jev and Claude over every evaluation frame (the only paid evaluation step).

python -m training.predict            # all tasks and sets
python -m training.predict --yes      # skip the cost confirmation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.cache import cache_for, confirm_cost, predict_cached
from training.evalsets import EVAL_SETS, load_eval
from training.llm import claude_classifier, estimate_claude_usd, jev_classifier
from training.paths import PREDS
from training.tasks import TASKS, labels_for
from training.train import latest_artifact, load_classic
from triage_core.classify import Prediction

MODEL_NAMES = ("classic", "jev", "claude")


def write_preds(
    task: str,
    set_name: str,
    model: str,
    preds: dict[str, Prediction],
    root: Path = PREDS,
) -> Path:
    path = root / f"{task}_{set_name}_{model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"complaint_id": cid, **asdict(p)}) + "\n" for cid, p in preds.items()),
        encoding="utf-8",
    )
    return path


def read_preds(task: str, set_name: str, model: str, root: Path = PREDS) -> dict[str, Prediction]:
    path = root / f"{task}_{set_name}_{model}.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        cid = data.pop("complaint_id")
        out[cid] = Prediction(**data)
    return out


def model_preds(task: str, set_name: str, root: Path = PREDS) -> dict[str, dict[str, Prediction]]:
    return {
        m: read_preds(task, set_name, m, root)
        for m in MODEL_NAMES
        if (root / f"{task}_{set_name}_{m}.jsonl").exists()
    }


def main(assume_yes: bool) -> None:
    classifiers = {
        task: {
            "classic": load_classic(latest_artifact(task)),
            "jev": jev_classifier(task),
            "claude": claude_classifier(task),
        }
        for task in TASKS
    }
    frames = {(t, s): load_eval(t, s) for t in TASKS for s in EVAL_SETS[t]}

    calls, usd = 0, 0.0
    for (task, _), frame in frames.items():
        for name in ("jev", "claude"):
            clf = classifiers[task][name]
            if clf is None:
                continue
            cache = cache_for(clf, labels_for(task))
            missing_ids = set(cache.missing(frame["complaint_id"].tolist()))
            texts = frame.loc[frame["complaint_id"].isin(missing_ids), "text"].tolist()
            calls += len(texts)
            if name == "claude":
                usd += estimate_claude_usd(texts, clf.model_id)
            else:
                usd += len(texts) * clf.usd_per_call
    confirm_cost(calls, usd, assume_yes)

    for (task, set_name), frame in frames.items():
        for name, clf in classifiers[task].items():
            if clf is None:
                continue
            preds = predict_cached(
                clf,
                cache_for(clf, labels_for(task)),
                frame["complaint_id"].tolist(),
                frame["text"].tolist(),
                labels_for(task),
                failed_path=PREDS / f"{task}_{set_name}_{name}_failed.jsonl",
            )
            path = write_preds(task, set_name, name, preds)
            print(f"{task}/{set_name}/{name}: {len(preds)} → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    main(parser.parse_args().yes)
