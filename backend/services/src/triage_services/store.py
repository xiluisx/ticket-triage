"""Tickets table (PK tenant_id, SK ticket_id = ULID, so a reverse query is newest first)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from boto3.dynamodb.conditions import Key

from triage_services.schemas import Ticket


def to_dynamo(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: to_dynamo(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_dynamo(v) for v in value]
    return value


def from_dynamo(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: from_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_dynamo(v) for v in value]
    return value


def create_table(dynamodb, name: str) -> None:
    """Same key schema as Terraform; used by tests and the local run."""
    dynamodb.create_table(
        TableName=name,
        KeySchema=[
            {"AttributeName": "tenant_id", "KeyType": "HASH"},
            {"AttributeName": "ticket_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "tenant_id", "AttributeType": "S"},
            {"AttributeName": "ticket_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class TicketStore:
    def __init__(self, table):
        self.table = table

    def put(self, ticket: Ticket) -> None:
        self.table.put_item(
            Item=to_dynamo(ticket.model_dump()),
            ConditionExpression="attribute_not_exists(ticket_id)",
        )

    def get(self, tenant_id: str, ticket_id: str) -> Ticket | None:
        item = self.table.get_item(Key={"tenant_id": tenant_id, "ticket_id": ticket_id}).get("Item")
        return Ticket.model_validate(from_dynamo(item)) if item else None

    def list_recent(self, tenant_id: str, limit: int = 200) -> list[Ticket]:
        resp = self.table.query(
            KeyConditionExpression=Key("tenant_id").eq(tenant_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [Ticket.model_validate(from_dynamo(i)) for i in resp["Items"]]
