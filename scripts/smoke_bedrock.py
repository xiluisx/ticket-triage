"""Smoke test: verify access to an Anthropic Claude Haiku model on Bedrock (us-east-1)."""

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REGION = "us-east-1"
PING = [{"role": "user", "content": [{"text": "Reply with the single word OK."}]}]


def candidate_models(bedrock) -> list[str]:
    """BEDROCK_MODEL_ID if set; otherwise every US Haiku inference profile, newest first."""
    override = os.environ.get("BEDROCK_MODEL_ID")
    if override:
        return [override]
    profiles = bedrock.list_inference_profiles()["inferenceProfileSummaries"]
    haiku = sorted(
        (
            p["inferenceProfileId"]
            for p in profiles
            if p["inferenceProfileId"].startswith("us.anthropic.")
            and "haiku" in p["inferenceProfileId"]
        ),
        reverse=True,
    )
    if not haiku:
        raise SystemExit("No Anthropic Haiku inference profile found in us-east-1")
    return haiku


def main() -> int:
    load_dotenv(ROOT / ".env")
    session = boto3.Session(region_name=REGION)
    runtime = session.client("bedrock-runtime")
    for model_id in candidate_models(session.client("bedrock")):
        try:
            resp = runtime.converse(
                modelId=model_id,
                messages=PING,
                inferenceConfig={"maxTokens": 5},
            )
        except ClientError as err:
            print(f"{model_id}: {err.response['Error']['Code']}")
            continue
        text = resp["output"]["message"]["content"][0]["text"]
        print(f"model={model_id} reply={text!r} usage={resp['usage']}")
        print(f"Set BEDROCK_MODEL_ID={model_id} in .env to pin it.")
        return 0
    print(
        "No Haiku model accessible. Bedrock console → Model catalog → Anthropic → "
        "submit use case details, then retry."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
