"""API payloads and the ticket item stored in DynamoDB."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Level = Literal["classic", "jev", "claude", "none"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TicketCreate(_Model):
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=50_000)


class TaskPrediction(_Model):
    label: str | None
    confidence: float
    level: Level
    model_version: str | None = None
    latency_ms: float
    cost_usd: float
    rationale: str | None = None
    skipped: list[str] = Field(default_factory=list)


class TicketPrediction(_Model):
    category: TaskPrediction
    priority: TaskPrediction


class Ticket(_Model):
    tenant_id: str
    ticket_id: str
    channel: Literal["email", "api"]
    received_at: str
    subject: str
    body_redacted: str
    pii_counts: dict[str, int] = Field(default_factory=dict)
    prediction: TicketPrediction
    final_category: str | None
    final_priority: str | None
    corrected: bool = False
    status: Literal["open", "in_progress", "resolved"] = "open"
    updated_at: str


class TicketList(_Model):
    items: list[Ticket]


class ErrorBody(_Model):
    code: str
    message: str


class ErrorResponse(_Model):
    error: ErrorBody
