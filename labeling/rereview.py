"""Blind re-review of golden-set priority with the binary definition.

uv run python labeling/rereview.py prepare
    → data/golden/rereview_import.json (Label Studio tasks, no Claude labels)
    → labeling/rereview_config.xml (priority only, binary definitions)
uv run python labeling/rereview.py apply [export.json]
    → updates datasets/golden/golden_v2.jsonl, writes data/golden/rereview_log.jsonl
"""

import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

from training.golden import apply_review, select_for_review
from training.jsonl import read_jsonl, write_jsonl
from training.paths import DISTILL, GOLDEN_V2, PRIORITIES
from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "golden"
N_CONTROL = 25
SEED = 42


def disagreement_ids() -> set[str]:
    return {r["complaint_id"] for r in read_jsonl(DISTILL / "gate_disagreements.jsonl")}


def failed_ids() -> set[str]:
    path = DISTILL / "gate_failed.jsonl"
    return {r["complaint_id"] for r in read_jsonl(path)} if path.exists() else set()


def config_xml(labels: LabelSet) -> str:
    choices = "\n".join(
        f'    <Choice value="{escape(lb.id)}" hint="{escape(lb.name)}: {escape(lb.description)}"/>'
        for lb in labels.labels
    )
    definitions = " | ".join(f"{lb.id}: {lb.description}" for lb in labels.labels)
    return "\n".join(
        [
            "<View>",
            f'  <Header value="{escape(definitions)}" size="6"/>',
            '  <Text name="text" value="$text"/>',
            '  <Choices name="priority" toName="text" choice="single" required="true">',
            choices,
            "  </Choices>",
            "</View>",
        ]
    )


def prepare() -> None:
    rows = read_jsonl(GOLDEN_V2)
    by_id = {r["complaint_id"]: r for r in rows}
    picked = select_for_review(
        list(by_id), disagreement_ids(), N_CONTROL, exclude=failed_ids(), seed=SEED
    )
    tasks = [{"data": {"complaint_id": cid, "text": by_id[cid]["text"]}} for cid in picked]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rereview_import.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (ROOT / "labeling" / "rereview_config.xml").write_text(
        config_xml(LabelSet.from_json(PRIORITIES)) + "\n", encoding="utf-8"
    )
    print(f"{len(tasks)} tasks → {OUT / 'rereview_import.json'}")
    print(f"config → {ROOT / 'labeling' / 'rereview_config.xml'}")


def apply(export_path: Path) -> None:
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    decisions = {}
    for task in exported:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if done:
            result = {r["from_name"]: r["value"]["choices"][0] for r in done[-1]["result"]}
            decisions[task["data"]["complaint_id"]] = result["priority"]
    disagreements = disagreement_ids()
    rows, log = apply_review(read_jsonl(GOLDEN_V2), decisions, disagreements)
    write_jsonl(GOLDEN_V2, rows)
    write_jsonl(OUT / "rereview_log.jsonl", log)
    flips = Counter((r["group"], r["before"] != r["after"]) for r in log)
    for group in ("disagreement", "control"):
        n = flips[(group, True)] + flips[(group, False)]
        print(f"{group}: {flips[(group, True)]}/{n} changed")
    print(f"golden v2 priority: {dict(Counter(r['priority'] for r in rows))}")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        default = OUT / "rereview_export.json"
        apply(Path(sys.argv[2]) if len(sys.argv) > 2 else default)
