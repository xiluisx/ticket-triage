import datetime as dt

from triage_core.ids import iso_z, new_ulid, ulid_at, ulid_from_message

T0 = dt.datetime(2026, 11, 24, 10, 0, 0, 123000, tzinfo=dt.UTC)


def test_ulid_shape():
    u = new_ulid(T0)
    assert len(u) == 26
    assert set(u) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_ulid_sorts_by_time():
    a = new_ulid(T0)
    b = new_ulid(T0 + dt.timedelta(milliseconds=1))
    assert a < b


def test_ulid_from_message_is_deterministic():
    a = ulid_from_message(T0, "msg-1")
    assert a == ulid_from_message(T0, "msg-1")
    assert a != ulid_from_message(T0, "msg-2")
    assert a[:10] == new_ulid(T0)[:10]  # same 48-bit timestamp prefix


def test_ulid_at_known_value():
    assert ulid_at(dt.datetime(1970, 1, 1, tzinfo=dt.UTC), bytes(10)) == "0" * 26


def test_iso_z():
    assert iso_z(T0) == "2026-11-24T10:00:00.123Z"
