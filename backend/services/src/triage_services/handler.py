"""Lambda entry point: Function URL (HTTP API v2 payload) -> Mangum -> FastAPI.

Nothing touches AWS at import time; models and clients load on the first request.
"""

from __future__ import annotations

import logging

import boto3
from mangum import Mangum

from triage_services.app import Deps, create_app
from triage_services.clients import TASKS, aws_classifiers, fake_classifiers
from triage_services.emf import EmfLogger
from triage_services.engine import build_engine
from triage_services.settings import Settings
from triage_services.store import TicketStore

logging.getLogger().setLevel(logging.INFO)


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or Settings.from_env()
    session = boto3.session.Session(region_name=settings.region)
    classifiers = (
        fake_classifiers() if settings.local else aws_classifiers(settings, session.client("ssm"))
    )
    return Deps(
        settings=settings,
        store=TicketStore(session.resource("dynamodb").Table(settings.tickets_table)),
        engines={task: build_engine(task, settings, classifiers) for task in TASKS},
        emf=EmfLogger(),
    )


app = create_app(build_deps)
handler = Mangum(app, lifespan="off")
