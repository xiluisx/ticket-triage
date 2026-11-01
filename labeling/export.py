"""Convert a Label Studio JSON export into datasets/golden/golden_v1.jsonl."""

import json
import sys
from pathlib import Path

from training.golden import export_annotations
from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    default = ROOT / "data" / "golden" / "labelstudio_export.json"
    export_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    rows = export_annotations(
        json.loads(export_path.read_text(encoding="utf-8")),
        LabelSet.from_json(ROOT / "datasets" / "categories_v1.json"),
        LabelSet.from_json(ROOT / "datasets" / "priorities_v1.json"),
    )
    out = ROOT / "datasets" / "golden" / "golden_v1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"{len(rows)} labeled rows → {out}")
