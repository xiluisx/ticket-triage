"""Download the CFPB narratives archive and build data/cfpb/complaints.parquet."""

import json
from pathlib import Path

from training.cfpb import build_dataset

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    stats = build_dataset(
        raw_dir=ROOT / "data" / "cfpb" / "archive",
        out_path=ROOT / "data" / "cfpb" / "complaints.parquet",
    )
    (ROOT / "data" / "cfpb" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
