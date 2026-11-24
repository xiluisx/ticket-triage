"""REST API (FastAPI). Every route needs the tenant API key in X-Api-Key.

Errors always use {"error": {"code", "message"}}. Ticket text is never logged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from triage_core.ids import iso_z, new_ulid, utc_now
from triage_core.pii import merge_counts, redact
from triage_services.emf import EmfLogger
from triage_services.engine import TaskEngine, classify, compose_text
from triage_services.schemas import (
    ErrorResponse,
    Ticket,
    TicketCreate,
    TicketList,
    TicketPrediction,
)
from triage_services.settings import Settings
from triage_services.store import TicketStore

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 8000  # same cut as training
HTTP_CODES = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "not_found"}
ERRORS = {s: {"model": ErrorResponse} for s in (401, 404, 422, 500)}


@dataclass
class Deps:
    settings: Settings
    store: TicketStore
    engines: dict[str, TaskEngine]
    emf: EmfLogger
    clock: Callable[[], dt.datetime] = field(default=utc_now)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def get_deps(request: Request) -> Deps:
    state = request.app.state
    if state.deps is None:  # cold start: load models and clients once per container
        state.deps = state.deps_factory()
    return state.deps


def authorized(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> Deps:
    deps = get_deps(request)
    expected = deps.settings.api_key_sha256
    given = hashlib.sha256((x_api_key or "").encode("utf-8")).hexdigest()
    if not x_api_key or not expected or not hmac.compare_digest(given, expected):
        raise ApiError(401, "unauthorized", "missing or invalid API key")
    return deps


Authorized = Annotated[Deps, Depends(authorized)]


def create_app(deps_factory: Callable[[], Deps]) -> FastAPI:
    app = FastAPI(
        title="Ticket Triage API",
        version="1.0.0",
        description="Classifies support tickets (category + priority) with a cost-aware "
        "cascade: classic model -> Jev -> Claude Haiku 4.5. PII is redacted before storage.",
    )
    app.state.deps_factory = deps_factory
    app.state.deps = None

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return error_response(422, "validation_error", msg[:500])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = HTTP_CODES.get(exc.status_code, "validation_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.error("unhandled %s on %s", type(exc).__name__, request.url.path)
        deps = request.app.state.deps
        if deps is not None:
            deps.emf.emit(
                {"event": "error", "path": request.url.path, "error": type(exc).__name__},
                {"Errors": 1},
            )
        return error_response(500, "internal_error", "internal error")

    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/tickets", status_code=201, response_model=Ticket, responses=ERRORS)
    def create_ticket(payload: TicketCreate, deps: Authorized) -> Ticket:
        subject = redact(payload.subject)
        body = redact(payload.body)
        body_text = body.text[:MAX_BODY_CHARS]
        pii_counts = merge_counts(subject.counts, body.counts)

        start = time.perf_counter()
        preds = classify(deps.engines, compose_text(subject.text, body_text))
        wall_ms = (time.perf_counter() - start) * 1000

        now = deps.clock()
        ticket = Ticket(
            tenant_id=deps.settings.tenant_id,
            ticket_id=new_ulid(now),
            channel="api",
            received_at=iso_z(now),
            subject=subject.text,
            body_redacted=body_text,
            pii_counts=pii_counts,
            prediction=TicketPrediction(category=preds["category"], priority=preds["priority"]),
            final_category=preds["category"].label,
            final_priority=preds["priority"].label,
            updated_at=iso_z(now),
        )
        deps.store.put(ticket)
        cost = sum(p.cost_usd for p in preds.values())
        deps.emf.emit(
            {
                "event": "ticket_classified",
                "tenant_id": ticket.tenant_id,
                "ticket_id": ticket.ticket_id,
                "channel": ticket.channel,
                **{
                    task: {
                        "label": p.label,
                        "confidence": p.confidence,
                        "level": p.level,
                        "model_version": p.model_version,
                        "skipped": p.skipped,
                    }
                    for task, p in preds.items()
                },
                "latency_ms": round(wall_ms, 1),
                "cost_usd": cost,
                "pii_counts": pii_counts,
                "body_chars": len(body_text),
            },
            {
                "TicketsProcessed": 1,
                "ClassificationLatency": wall_ms,
                "ClassificationCost": cost,
                "PiiRedacted": sum(pii_counts.values()),
                "Unclassified": int(any(p.level == "none" for p in preds.values())),
            },
        )
        return ticket

    @app.get("/v1/tickets", response_model=TicketList, responses=ERRORS)
    def list_tickets(
        deps: Authorized, limit: Annotated[int, Query(ge=1, le=200)] = 200
    ) -> TicketList:
        return TicketList(items=deps.store.list_recent(deps.settings.tenant_id, limit))

    @app.get("/v1/tickets/{ticket_id}", response_model=Ticket, responses=ERRORS)
    def get_ticket(ticket_id: str, deps: Authorized) -> Ticket:
        ticket = deps.store.get(deps.settings.tenant_id, ticket_id)
        if ticket is None:
            raise ApiError(404, "not_found", "ticket not found")
        return ticket

    return app
