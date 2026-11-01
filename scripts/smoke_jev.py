"""Smoke test: classify 5 CFPB narratives with Jev (TypeSafe System One, Choice primitive).

Reads TYPESAFE_API_KEY from .env. Exits non-zero on any failure.
"""

import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from triage_core.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = "Which support team should handle the customer complaint in `ticket`?"


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("TYPESAFE_API_KEY missing in .env")
        return 1

    labels = LabelSet.from_json(ROOT / "datasets" / "categories_v1.json")
    df = pd.read_parquet(
        ROOT / "data" / "cfpb" / "complaints.parquet", columns=["narrative", "product"]
    )
    sample = df.sample(5, random_state=7)

    with httpx.Client(timeout=10, headers={"Authorization": f"Bearer {api_key}"}) as client:
        for narrative, product in sample.itertuples(index=False):
            body = {
                "model": "jev-latest",
                "state": {"ticket": narrative[:4000]},
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": labels.descriptions(),
                    }
                },
            }
            start = time.perf_counter()
            resp = client.post(API_URL, json=body)
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.text[:500]}")
                return 1
            data = resp.json()
            answer = data["answers"]["category"]
            top = sorted(answer["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(f"product={product!r}")
            print(f"  jev={answer['choice']} conf={answer['confidence']:.2f} top3={top}")
            print(f"  latency={latency_ms:.0f}ms model={data['model']} usage={data['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
