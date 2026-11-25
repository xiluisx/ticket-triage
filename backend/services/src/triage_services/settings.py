"""Runtime configuration from environment variables (Lambda env or local run)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from triage_core.llm_config import DEFAULT_CLAUDE_MODEL

# In the repo, backend/services/src/triage_services/settings.py -> root is parents[4].
# Installed in the container the path is shorter; there MODELS_DIR and DATASETS_DIR are set.
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[4] if len(_HERE.parents) > 4 else _HERE.parent


@dataclass(frozen=True)
class Settings:
    env: str = "prod"
    region: str = "us-east-1"
    tenant_id: str = "cfpb-demo"
    tickets_table: str = "ticket-triage-tickets"
    api_key_sha256: str = ""
    models_dir: Path = REPO_ROOT / "models"
    datasets_dir: Path = REPO_ROOT / "datasets"
    jev_api_key_param: str = "/ticket-triage/typesafe-api-key"
    claude_model_id: str = DEFAULT_CLAUDE_MODEL
    jev_usd_per_call: float = 0.0

    @property
    def local(self) -> bool:
        return self.env == "local"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Settings:
        d = cls()
        return cls(
            env=environ.get("TRIAGE_ENV", d.env),
            region=environ.get("AWS_REGION", d.region),
            tenant_id=environ.get("TENANT_ID", d.tenant_id),
            tickets_table=environ.get("TICKETS_TABLE", d.tickets_table),
            api_key_sha256=environ.get("API_KEY_SHA256", d.api_key_sha256),
            models_dir=Path(environ.get("MODELS_DIR", d.models_dir)),
            datasets_dir=Path(environ.get("DATASETS_DIR", d.datasets_dir)),
            jev_api_key_param=environ.get("JEV_API_KEY_PARAM", d.jev_api_key_param),
            claude_model_id=environ.get("CLAUDE_MODEL_ID", d.claude_model_id),
            jev_usd_per_call=float(environ.get("JEV_USD_PER_CALL", d.jev_usd_per_call)),
        )
