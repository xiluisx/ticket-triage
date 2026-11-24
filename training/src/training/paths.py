"""Repository paths shared by training scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ROOT / "datasets"
DATA = ROOT / "data"
CFPB = DATA / "cfpb" / "complaints.parquet"
SPLITS = DATA / "splits"
CACHE = DATA / "cache"
DISTILL = DATA / "distill"
EVAL = DATA / "eval"
PREDS = DATA / "preds"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"
CATEGORIES = DATASETS / "categories_v1.json"
PRIORITIES = DATASETS / "priorities_v2.json"
GOLDEN_V1 = DATASETS / "golden" / "golden_v1.jsonl"
GOLDEN_V2 = DATASETS / "golden" / "golden_v2.jsonl"
DISTILLED = DISTILL / "claude_priority_v1.jsonl"
