import pytest

from triage_core.pii import luhn_ok, merge_counts, redact


@pytest.mark.parametrize(
    ("text", "expected", "counts"),
    [
        ("mail me at john.doe+x@mail.example.com now", "mail me at XXXX now", {"email": 1}),
        ("SSN 123-45-6789 on file", "SSN XXXX on file", {"ssn": 1}),
        ("my ssn is 123456789.", "my ssn is XXXX.", {"ssn": 1}),
        ("card 4111 1111 1111 1111 charged", "card XXXX charged", {"card": 1}),
        ("card 4111-1111-1111-1111", "card XXXX", {"card": 1}),
        ("account number 12345678 frozen", "account number XXXX frozen", {"account": 1}),
        ("acct #: 9876543210", "acct #: XXXX", {"account": 1}),
        ("loan number 555123", "loan number XXXX", {"account": 1}),
        ("call (555) 123-4567 today", "call XXXX today", {"phone": 1}),
        ("call +1 555 123 4567", "call XXXX", {"phone": 1}),
        ("call 555.123.4567", "call XXXX", {"phone": 1}),
    ],
)
def test_redacts_each_type(text, expected, counts):
    out = redact(text)
    assert out.text == expected
    assert out.counts == counts


@pytest.mark.parametrize(
    "text",
    [
        "I was charged $1,234.56 on XX/XX/XXXX",
        "paid {$100.00} on 11/24/2026",
        "my balance is 12345678 dollars",  # digits without an account keyword
        "card 4111 1111 1111 1112",  # fails Luhn
        "XXXX XXXX already masked by CFPB",
        "reference 2024",
    ],
)
def test_leaves_non_pii_alone(text):
    out = redact(text)
    assert out.text == text
    assert out.counts == {}


def test_counts_multiple():
    out = redact("a@b.co and c@d.co, phone 555-123-4567")
    assert out.counts == {"email": 2, "phone": 1}
    assert "@" not in out.text


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("1234")


def test_merge_counts():
    assert merge_counts({"email": 1}, {"email": 2, "card": 1}) == {"email": 3, "card": 1}
