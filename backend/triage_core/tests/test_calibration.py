import pytest

from triage_core.calibration import apply_temperature


def test_temperature_one_is_identity():
    probs = {"a": 0.8, "b": 0.2}
    assert apply_temperature(probs, 1.0) == probs


def test_temperature_above_one_softens_and_keeps_order():
    out = apply_temperature({"a": 0.9, "b": 0.1}, 2.0)
    assert out["a"] < 0.9
    assert out["a"] > out["b"]
    assert sum(out.values()) == pytest.approx(1.0)


def test_temperature_below_one_sharpens():
    assert apply_temperature({"a": 0.6, "b": 0.4}, 0.5)["a"] > 0.6
