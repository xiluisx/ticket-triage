"""Append-only JSONL cache of predictions so re-runs never pay twice for the same call."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from training.jsonl import write_jsonl
from training.paths import CACHE
from triage_core.classify import Classifier, Prediction
from triage_core.labels import LabelSet


class RunFailedError(RuntimeError):
    """More rows failed than the allowed rate."""


class PredictionCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rows: dict[str, Prediction] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    cid = data.pop("complaint_id")
                    self._rows[cid] = Prediction(**data)

    def __contains__(self, cid: str) -> bool:
        return cid in self._rows

    def get(self, cid: str) -> Prediction | None:
        return self._rows.get(cid)

    def put(self, cid: str, pred: Prediction) -> None:
        with self._lock:
            self._rows[cid] = pred
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"complaint_id": cid, **asdict(pred)}, ensure_ascii=False))
                f.write("\n")

    def missing(self, ids: list[str]) -> list[str]:
        return [cid for cid in ids if cid not in self._rows]


def cache_for(clf: Classifier, labels: LabelSet, root: Path = CACHE) -> PredictionCache:
    """One cache file per classifier config and label set (labels change prompts and criteria)."""
    return PredictionCache(root / f"{clf.cache_key}-{labels.labels_hash()}.jsonl")


def predict_cached(
    clf: Classifier,
    cache: PredictionCache,
    ids: list[str],
    texts: list[str],
    labels: LabelSet,
    workers: int = 4,
    max_fail_rate: float = 0.02,
    failed_path: Path | None = None,
) -> dict[str, Prediction]:
    todo = [(cid, text) for cid, text in zip(ids, texts, strict=True) if cid not in cache]

    def work(cid: str, text: str) -> None:
        start = time.perf_counter()
        pred = clf.predict([text], labels)[0]
        if pred.latency_ms == 0.0:
            pred = replace(pred, latency_ms=(time.perf_counter() - start) * 1000)
        cache.put(cid, pred)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, cid, text): cid for cid, text in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as err:  # noqa: BLE001 - every failure is recorded
                failures.append(
                    {"complaint_id": futures[future], "error": f"{type(err).__name__}: {err}"}
                )
            if i % 250 == 0:
                print(f"  {clf.name}: {i}/{len(todo)}")
    if failures and failed_path:
        write_jsonl(failed_path, failures)
    if len(failures) > max_fail_rate * len(ids):
        raise RunFailedError(f"{len(failures)}/{len(ids)} rows failed ({clf.name})")
    return {cid: cache.get(cid) for cid in ids if cid in cache}


def confirm_cost(
    n_calls: int,
    usd: float,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
) -> None:
    print(f"{n_calls} uncached paid calls, ~${usd:.2f} estimated")
    if n_calls == 0 or assume_yes:
        return
    if input_fn("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted by user")
