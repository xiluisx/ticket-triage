import dataclasses
import datetime as dt
import hashlib
import io
from types import SimpleNamespace

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from triage_services.app import Deps, create_app
from triage_services.clients import TASKS, fake_classifiers
from triage_services.emf import EmfLogger
from triage_services.engine import build_engine
from triage_services.settings import Settings
from triage_services.store import TicketStore, create_table

API_KEY = "tt_cfpb-demo_test-secret"
T0 = dt.datetime(2026, 11, 24, 10, 0, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def fake_aws_env(monkeypatch):
    """Never touch a real account: drop the project profile exported by the Makefile."""
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(
        Settings(),
        env="local",
        api_key_sha256=hashlib.sha256(API_KEY.encode()).hexdigest(),
        models_dir=tmp_path / "no-models",  # LLM-only cascade unless a test adds models
    )


@pytest.fixture
def table(settings):
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        create_table(dynamodb, settings.tickets_table)
        yield dynamodb.Table(settings.tickets_table)


@pytest.fixture
def api(settings, table):
    out = io.StringIO()
    clfs = fake_classifiers()
    deps = Deps(
        settings=settings,
        store=TicketStore(table),
        engines={t: build_engine(t, settings, clfs) for t in TASKS},
        emf=EmfLogger(out),
        clock=lambda: T0,
    )
    client = TestClient(create_app(lambda: deps), raise_server_exceptions=False)
    return SimpleNamespace(client=client, deps=deps, emf=out, headers={"X-Api-Key": API_KEY})
