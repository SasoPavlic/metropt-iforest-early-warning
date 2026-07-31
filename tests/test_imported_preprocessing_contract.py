import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from pdm_eval.detectors import imported_recurrent_autoencoder_detector as imported
from pdm_eval.manifest import _load_cycle_manifest, _resolve_manifest_cycle


def _base_meta(feature_names: list[str], preprocessing_contract: dict) -> dict:
    return {
        "schema_version": "2.0",
        "contract_version": "2.0",
        "feature_contract": {
            "rolling_feature_names": feature_names,
            "feature_hash": imported._feature_hash(feature_names),
            "n_features": len(feature_names),
        },
        "preprocessing_contract": preprocessing_contract,
        "sequence_contract": {
            "seq_len": 5,
            "stride": 1,
            "cross_gap_windows_allowed": False,
        },
        "split_contract": {},
    }


def _binary_contract_fixture() -> tuple[dict, StandardScaler]:
    feature_names = ["TP2__mean", "COMP__mean", "COMP__max"]
    values = np.asarray(
        [[float(index), 0.0, 0.0] for index in range(20)] + [[20.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    scaler = StandardScaler().fit(values)
    passthrough_indices = [1, 2]
    scaler.mean_[passthrough_indices] = 0.0
    scaler.scale_[passthrough_indices] = 1.0
    scaler.var_[passthrough_indices] = 1.0
    scaler.nianetvae_preprocessing_policy_ = "binary_passthrough_v1"
    scaler.nianetvae_preprocessing_policy_version_ = "1.0"
    scaler.nianetvae_passthrough_indices_ = np.asarray(
        passthrough_indices, dtype=np.int64
    )

    payload = {
        "policy": "binary_passthrough_v1",
        "policy_version": "1.0",
        "behavior": "continuous_derived_standardized_binary_derived_passthrough",
        "preserves_feature_order": True,
        "preserves_feature_count": True,
        "configured_binary_feature_names": ["COMP"],
        "matched_binary_feature_names": ["COMP"],
        "binary_derived_feature_indices": passthrough_indices,
        "binary_derived_feature_names": ["COMP__mean", "COMP__max"],
        "binary_derived_feature_count": 2,
        "applied_binary_feature_names": ["COMP"],
        "passthrough_feature_indices": passthrough_indices,
        "passthrough_feature_names": ["COMP__mean", "COMP__max"],
        "passthrough_feature_count": 2,
        "standardized_feature_indices": [0],
        "standardized_feature_count": 1,
    }
    preprocessing_contract = {
        **payload,
        "contract_hash": imported._json_hash(payload),
        "scaler_file": "scaler.joblib",
        "scaler_feature_count": len(feature_names),
        "scaler_hash": imported._json_hash(imported._scaler_state_payload(scaler)),
    }
    meta = _base_meta(feature_names, preprocessing_contract)
    meta["feature_contract"].update(
        {
            "binary_base_feature_names": ["COMP"],
            "binary_derived_feature_indices": passthrough_indices,
            "binary_derived_feature_names": ["COMP__mean", "COMP__max"],
        }
    )
    return meta, scaler


def _rehash_preprocessing_contract(meta: dict) -> None:
    contract = meta["preprocessing_contract"]
    contract["contract_hash"] = imported._json_hash(
        imported._preprocessing_contract_hash_payload(contract)
    )


def _validate_detector_contract(meta: dict, scaler: StandardScaler):
    detector = imported.ImportedRecurrentAutoencoderDetector.__new__(
        imported.ImportedRecurrentAutoencoderDetector
    )
    detector.meta = meta
    detector.cfg = SimpleNamespace(scaler_path="provided-scaler.joblib")
    detector._validate_contract_metadata()
    detector.feature_contract = meta["feature_contract"]
    detector.preprocessing_contract = meta["preprocessing_contract"]
    detector.input_dim = len(meta["feature_contract"]["rolling_feature_names"])
    detector.scaler = scaler
    detector._validate_scaler_contract()
    return detector


def test_missing_policy_remains_legacy_standard_scaler_compatible() -> None:
    feature_names = ["f0", "f1", "f2"]
    scaler = StandardScaler().fit(np.arange(30, dtype=float).reshape(10, 3))
    preprocessing_contract = {
        "scaler_file": "scaler.joblib",
        "scaler_feature_count": 3,
        "scaler_hash": imported._json_hash(imported._scaler_state_payload(scaler)),
    }

    detector = _validate_detector_contract(
        _base_meta(feature_names, preprocessing_contract), scaler
    )

    assert detector.preprocessing_policy == "standard_scaler_v1"
    assert detector.preprocessing_policy_version == "1.0"
    assert detector.preprocessing_policy_explicit is False


def test_binary_preprocessing_contract_and_scaler_are_accepted() -> None:
    meta, scaler = _binary_contract_fixture()

    detector = _validate_detector_contract(meta, scaler)

    assert detector.preprocessing_policy == "binary_passthrough_v1"
    assert detector.preprocessing_policy_explicit is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy", "unknown_v1", "Unsupported imported preprocessing policy"),
        ("policy_version", "2.0", "policy version"),
        ("contract_hash", "bad-hash", "contract hash"),
    ],
)
def test_binary_preprocessing_rejects_policy_version_and_hash_mismatch(
    field: str,
    value: str,
    message: str,
) -> None:
    meta, scaler = _binary_contract_fixture()
    meta["preprocessing_contract"][field] = value

    with pytest.raises(imported.ImportedArtifactContractError, match=message):
        _validate_detector_contract(meta, scaler)


def test_binary_preprocessing_rejects_passthrough_name_order_mismatch() -> None:
    meta, scaler = _binary_contract_fixture()
    meta["preprocessing_contract"]["passthrough_feature_names"] = [
        "COMP__max",
        "COMP__mean",
    ]
    _rehash_preprocessing_contract(meta)

    with pytest.raises(
        imported.ImportedArtifactContractError,
        match="passthrough feature names",
    ):
        _validate_detector_contract(meta, scaler)


def test_binary_preprocessing_rejects_scaler_custom_attribute_mismatch() -> None:
    meta, scaler = _binary_contract_fixture()
    scaler.nianetvae_passthrough_indices_ = np.asarray([1], dtype=np.int64)

    with pytest.raises(
        imported.ImportedArtifactContractError,
        match="scaler passthrough indices",
    ):
        _validate_detector_contract(meta, scaler)


def test_binary_preprocessing_rejects_non_identity_scaler_state() -> None:
    meta, scaler = _binary_contract_fixture()
    scaler.scale_[1] = 0.01

    with pytest.raises(
        imported.ImportedArtifactContractError,
        match="not identity",
    ):
        _validate_detector_contract(meta, scaler)


def _write_manifest_artifacts(
    tmp_path: Path,
    *,
    metadata_contract: dict,
) -> tuple[Path, Path, Path]:
    cycle_dir = tmp_path / "cycle_00"
    cycle_dir.mkdir(parents=True)
    model_path = cycle_dir / "model.pt"
    meta_path = cycle_dir / "model_meta.json"
    scaler_path = cycle_dir / "scaler.joblib"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    meta_path.write_text(
        json.dumps({"preprocessing_contract": metadata_contract}),
        encoding="utf-8",
    )
    return model_path, meta_path, scaler_path


def test_manifest_binary_summary_is_validated_against_model_metadata(
    tmp_path: Path,
) -> None:
    model_path, meta_path, scaler_path = _write_manifest_artifacts(
        tmp_path,
        metadata_contract={
            "policy": "binary_passthrough_v1",
            "policy_version": "1.0",
            "contract_hash": "contract-hash",
            "passthrough_feature_count": 48,
        },
    )
    manifest_path = tmp_path / "cycle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "contract_version": "2.0",
                "preprocessing_policy": "binary_passthrough_v1",
                "observed_preprocessing_policies": ["binary_passthrough_v1"],
                "cycles": {
                    "00": {
                        "status": "trained",
                        "contract_version": "2.0",
                        "cycle_id": 0,
                        "model_path": str(model_path),
                        "meta_path": str(meta_path),
                        "scaler_path": str(scaler_path),
                        "preprocessing_policy": "binary_passthrough_v1",
                        "preprocessing_policy_version": "1.0",
                        "preprocessing_contract_hash": "contract-hash",
                        "binary_passthrough_feature_count": 48,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest, manifest_dir = _load_cycle_manifest(str(manifest_path))
    resolved = _resolve_manifest_cycle(
        manifest=manifest,
        manifest_dir=manifest_dir,
        cycle_id=0,
        strict=True,
    )

    assert resolved is not None
    assert resolved["preprocessing_policy"] == "binary_passthrough_v1"
    assert resolved["binary_passthrough_feature_count"] == 48

    manifest["cycles"]["00"]["preprocessing_contract_hash"] = "wrong-hash"
    with pytest.raises(ValueError, match="Manifest/model metadata preprocessing mismatch"):
        _resolve_manifest_cycle(
            manifest=manifest,
            manifest_dir=manifest_dir,
            cycle_id=0,
            strict=True,
        )


def test_manifest_observed_policy_summary_must_match_trained_cycles(
    tmp_path: Path,
) -> None:
    model_path, meta_path, scaler_path = _write_manifest_artifacts(
        tmp_path,
        metadata_contract={},
    )
    manifest_path = tmp_path / "cycle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "contract_version": "2.0",
                "observed_preprocessing_policies": ["binary_passthrough_v1"],
                "cycles": {
                    "00": {
                        "status": "trained",
                        "contract_version": "2.0",
                        "cycle_id": 0,
                        "model_path": str(model_path),
                        "meta_path": str(meta_path),
                        "scaler_path": str(scaler_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observed_preprocessing_policies"):
        _load_cycle_manifest(str(manifest_path))
