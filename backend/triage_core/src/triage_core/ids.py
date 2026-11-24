"""Ticket ids (ULID: 48-bit ms timestamp + 80 bits, Crockford base32) and UTC timestamps."""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def ulid_at(when: dt.datetime, randomness: bytes) -> str:
    if len(randomness) != 10:
        raise ValueError("ULID randomness must be 10 bytes")
    ms = int(when.timestamp() * 1000)
    value = (ms << 80) | int.from_bytes(randomness, "big")
    chars = []
    for _ in range(26):
        value, rem = divmod(value, 32)
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars))


def new_ulid(when: dt.datetime | None = None) -> str:
    return ulid_at(when or utc_now(), secrets.token_bytes(10))


def ulid_from_message(received_at: dt.datetime, message_id: str) -> str:
    """Idempotent id for e-mail: SES retries of the same message map to the same ticket."""
    return ulid_at(received_at, hashlib.sha256(message_id.encode("utf-8")).digest()[:10])


def iso_z(when: dt.datetime) -> str:
    when = when.astimezone(dt.UTC)
    return f"{when:%Y-%m-%dT%H:%M:%S}.{when.microsecond // 1000:03d}Z"
