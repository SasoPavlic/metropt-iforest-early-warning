import pytest

from pdm_eval.visualization.plots import _risk_alarm_label


def test_risk_alarm_label_uses_authoritative_evaluation_coverage() -> None:
    label = _risk_alarm_label(0.6, 2.0 / 3.0)

    assert label == "Risk alarm (≥ θ=0.60, 66.7% evaluation coverage)"


def test_risk_alarm_label_omits_coverage_when_metric_is_unavailable() -> None:
    assert _risk_alarm_label(0.6, None) == "Risk alarm (≥ θ=0.60)"
    assert _risk_alarm_label(None, 0.5) == "Risk alarm"


@pytest.mark.parametrize("coverage", [-0.1, 1.1, float("nan"), float("inf")])
def test_risk_alarm_label_rejects_invalid_evaluation_coverage(coverage: float) -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        _risk_alarm_label(0.6, coverage)
