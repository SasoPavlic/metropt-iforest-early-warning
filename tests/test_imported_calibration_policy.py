import pandas as pd
import pytest

from pdm_eval import pipeline as metropt_main


def _calibration_fixture():
    index = pd.date_range("2020-01-01", periods=14, freq="1min")
    frame = pd.DataFrame({"feature": range(len(index))}, index=index)
    baseline = pd.Series(False, index=index)
    baseline.iloc[:6] = True
    local = pd.Series(False, index=index)
    local.iloc[10:] = True
    return frame, baseline | local, baseline, local


def _combined_index(segments):
    return pd.concat(segments).index


def test_full_baseline_calibration_preserves_previous_composition():
    frame, train_mask, baseline, _local = _calibration_fixture()

    segments, metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=train_mask,
        initial_baseline_mask=baseline,
        requested_cycle_id=1,
        policy="full_baseline",
    )

    assert _combined_index(segments).equals(frame.index[train_mask])
    assert [len(segment) for segment in segments] == [6, 4]
    assert metadata == {
        "configured_policy": "full_baseline",
        "effective_policy": "full_baseline",
        "selection": "all_available_rows",
        "baseline_rows_available": 6,
        "local_rows_available": 4,
        "current_local_rows_available": 4,
        "fallback_local_rows_available": 0,
        "local_source": "current_train",
        "local_reference_cycle_id": 1,
        "baseline_rows_used": 6,
        "local_rows_used": 4,
        "calibration_rows": 10,
        "calibration_segment_count": 2,
        "calibration_segment_sources": ["baseline", "local"],
    }


def test_balanced_calibration_is_equal_chronological_and_deterministic():
    frame, train_mask, baseline, local = _calibration_fixture()

    first_segments, first_metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=train_mask,
        initial_baseline_mask=baseline,
        requested_cycle_id=1,
        policy="balanced_baseline_local",
    )
    second_segments, second_metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=train_mask,
        initial_baseline_mask=baseline,
        requested_cycle_id=1,
        policy="balanced_baseline_local",
    )

    expected = baseline.copy()
    expected.iloc[:2] = False
    expected |= local
    assert _combined_index(first_segments).equals(frame.index[expected])
    assert _combined_index(second_segments).equals(_combined_index(first_segments))
    assert first_metadata == second_metadata
    assert first_metadata["selection"] == "chronological_tail_equal_rows"
    assert first_metadata["baseline_rows_used"] == 4
    assert first_metadata["local_rows_used"] == 4
    assert first_metadata["calibration_segment_sources"] == ["baseline", "local"]


def test_local_only_calibration_excludes_baseline_for_nonzero_cycle():
    frame, train_mask, baseline, local = _calibration_fixture()

    segments, metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=train_mask,
        initial_baseline_mask=baseline,
        requested_cycle_id=2,
        policy="local_only",
    )

    assert _combined_index(segments).equals(frame.index[local])
    assert metadata["effective_policy"] == "local_only"
    assert metadata["baseline_rows_used"] == 0
    assert metadata["local_rows_used"] == 4


@pytest.mark.parametrize(
    "configured_policy",
    ["full_baseline", "balanced_baseline_local", "local_only"],
)
def test_cycle_zero_is_always_full_baseline_only(configured_policy):
    frame, train_mask, baseline, _local = _calibration_fixture()

    segments, metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=train_mask,
        initial_baseline_mask=baseline,
        requested_cycle_id=0,
        policy=configured_policy,
    )

    assert _combined_index(segments).equals(frame.index[baseline])
    assert metadata["configured_policy"] == configured_policy
    assert metadata["effective_policy"] == "baseline_only_cycle0"
    assert metadata["baseline_rows_used"] == 6
    assert metadata["local_rows_used"] == 0


@pytest.mark.parametrize("policy", ["balanced_baseline_local", "local_only"])
def test_local_dependent_policy_fails_when_cycle_has_no_local_rows(policy):
    frame, _train_mask, baseline, _local = _calibration_fixture()

    with pytest.raises(ValueError, match="requires .*local rows"):
        metropt_main._build_imported_calibration_segments(
            X=frame,
            train_mask=baseline,
            initial_baseline_mask=baseline,
            requested_cycle_id=1,
            policy=policy,
        )


@pytest.mark.parametrize(
    ("policy", "expected_selection", "expected_baseline_rows"),
    [
        (
            "balanced_baseline_local",
            "chronological_tail_equal_rows_with_carried_forward_local",
            4,
        ),
        ("local_only", "all_carried_forward_local_rows", 0),
    ],
)
def test_local_dependent_policy_carries_forward_latest_audited_local_block(
    policy,
    expected_selection,
    expected_baseline_rows,
):
    frame, _train_mask, baseline, previous_local = _calibration_fixture()

    segments, metadata = metropt_main._build_imported_calibration_segments(
        X=frame,
        train_mask=baseline,
        initial_baseline_mask=baseline,
        requested_cycle_id=18,
        policy=policy,
        fallback_local_mask=previous_local,
        fallback_local_cycle_id=17,
    )

    assert metadata["local_source"] == "carried_forward_latest_available"
    assert metadata["local_reference_cycle_id"] == 17
    assert metadata["current_local_rows_available"] == 0
    assert metadata["fallback_local_rows_available"] == 4
    assert metadata["local_rows_used"] == 4
    assert metadata["baseline_rows_used"] == expected_baseline_rows
    assert metadata["selection"] == expected_selection
    assert len(segments) == (2 if policy == "balanced_baseline_local" else 1)


def test_unknown_imported_calibration_policy_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported IMPORTED_CALIBRATION_COMPOSITION_POLICY",
    ):
        metropt_main._normalize_imported_calibration_policy("random_mix")
