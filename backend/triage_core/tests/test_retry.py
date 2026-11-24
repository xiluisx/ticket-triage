import pytest

from triage_core.retry import RetryableError, call_with_retries


def flaky(failures: int):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise RetryableError("busy")
        return "ok"

    return fn, calls


def is_retryable(err):
    return isinstance(err, RetryableError)


def test_retries_then_succeeds_with_exponential_delays():
    fn, calls = flaky(2)
    sleeps = []
    out = call_with_retries(fn, is_retryable=is_retryable, sleep=sleeps.append, jitter=lambda: 0)
    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_attempts():
    fn, calls = flaky(10)
    with pytest.raises(RetryableError):
        call_with_retries(fn, is_retryable=is_retryable, attempts=3, sleep=lambda s: None)
    assert calls["n"] == 3


def test_non_retryable_raises_immediately():
    def fn():
        raise ValueError("bad request")

    sleeps = []
    with pytest.raises(ValueError):
        call_with_retries(fn, is_retryable=is_retryable, sleep=sleeps.append)
    assert sleeps == []
