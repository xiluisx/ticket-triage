"""Write labeling/config.xml from the versioned label files."""

from pathlib import Path

from training.golden import label_studio_config
from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    xml = label_studio_config(
        LabelSet.from_json(ROOT / "datasets" / "categories_v1.json"),
        LabelSet.from_json(ROOT / "datasets" / "priorities_v1.json"),
    )
    (ROOT / "labeling" / "config.xml").write_text(xml + "\n", encoding="utf-8")
    print("labeling/config.xml written")
