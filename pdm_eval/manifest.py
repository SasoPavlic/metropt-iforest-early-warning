"""Manifest loading and cycle-artifact resolution for imported NiaNetVAE models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .detectors.imported_recurrent_autoencoder_detector import (
    ARTIFACT_CONTRACT_VERSION,
    BINARY_AWARE_PREPROCESSING_POLICY,
    LEGACY_PREPROCESSING_POLICY,
    PREPROCESSING_POLICY_VERSION,
    SUPPORTED_PREPROCESSING_POLICIES,
)


def _cycle_key(cycle_id: int) -> str:
    return f"{int(cycle_id):02d}"


def _manifest_preprocessing_summary(entry: dict) -> dict:
    policy = str(
        entry.get("preprocessing_policy") or LEGACY_PREPROCESSING_POLICY
    ).strip().lower()
    if policy not in SUPPORTED_PREPROCESSING_POLICIES:
        raise ValueError(f"Unsupported manifest preprocessing_policy={policy!r}.")
    version = str(
        entry.get("preprocessing_policy_version") or PREPROCESSING_POLICY_VERSION
    )
    if version != PREPROCESSING_POLICY_VERSION:
        raise ValueError(
            "Unsupported manifest preprocessing policy version: "
            f"policy={policy!r}, version={version!r}."
        )
    contract_hash = entry.get("preprocessing_contract_hash")
    passthrough_count = int(entry.get("binary_passthrough_feature_count") or 0)
    if policy == BINARY_AWARE_PREPROCESSING_POLICY:
        if not contract_hash:
            raise ValueError(
                "Binary-aware manifest cycle is missing preprocessing_contract_hash."
            )
        if passthrough_count < 1:
            raise ValueError(
                "Binary-aware manifest cycle must declare a positive "
                "binary_passthrough_feature_count."
            )
    elif passthrough_count != 0:
        raise ValueError(
            "Legacy manifest preprocessing policy cannot declare binary passthrough features."
        )
    return {
        "policy": policy,
        "version": version,
        "contract_hash": contract_hash,
        "passthrough_count": passthrough_count,
    }


def _metadata_preprocessing_summary(meta_path: Path) -> dict:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse imported model metadata: {meta_path}") from exc
    preprocessing_contract = metadata.get("preprocessing_contract")
    if not isinstance(preprocessing_contract, dict):
        raise ValueError(
            f"Imported model metadata is missing preprocessing_contract: {meta_path}"
        )
    return _manifest_preprocessing_summary(
        {
            "preprocessing_policy": preprocessing_contract.get("policy"),
            "preprocessing_policy_version": preprocessing_contract.get(
                "policy_version"
            ),
            "preprocessing_contract_hash": preprocessing_contract.get("contract_hash"),
            "binary_passthrough_feature_count": preprocessing_contract.get(
                "passthrough_feature_count", 0
            ),
        }
    )


def _load_cycle_manifest(path: str) -> tuple[dict, Path]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"PER_MAINT_MODEL_MANIFEST_PATH not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must use contract v2 "
            f"(schema_version={ARTIFACT_CONTRACT_VERSION!r}); "
            f"got schema_version={payload.get('schema_version')!r}."
        )
    if str(payload.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must declare "
            f"contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
            f"got contract_version={payload.get('contract_version')!r}."
        )
    if "cycles" not in payload or not isinstance(payload["cycles"], dict):
        raise ValueError(f"Invalid cycle manifest format at {p}: missing 'cycles' object.")
    top_level_policy = str(
        payload.get("preprocessing_policy") or LEGACY_PREPROCESSING_POLICY
    ).strip().lower()
    if top_level_policy not in SUPPORTED_PREPROCESSING_POLICIES:
        raise ValueError(
            f"Unsupported top-level manifest preprocessing_policy={top_level_policy!r}."
        )
    observed_policies: set[str] = set()
    for key, entry in payload["cycles"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cycle manifest format at {p}: cycle {key} entry is not an object.")
        status = str(entry.get("status", "")).strip().lower()
        if status == "trained":
            if str(entry.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
                raise ValueError(
                    f"Cycle {key} must use contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
                    f"got {entry.get('contract_version')!r}."
                )
            missing = [field for field in ("model_path", "meta_path", "scaler_path") if not entry.get(field)]
            if missing:
                raise ValueError(
                    f"Cycle {key} status=trained is missing required v2 fields: {', '.join(missing)}."
                )
            summary = _manifest_preprocessing_summary(entry)
            observed_policies.add(summary["policy"])
        elif status == "alias":
            if entry.get("alias_to") is None:
                raise ValueError(f"Cycle {key} status=alias is missing alias_to.")
        elif status != "missing":
            raise ValueError(f"Cycle {key} has unsupported status={status!r}.")
    declared_observed = payload.get("observed_preprocessing_policies")
    if declared_observed is not None:
        if not isinstance(declared_observed, list):
            raise ValueError(
                "Manifest observed_preprocessing_policies must be a list when present."
            )
        normalized_declared = sorted(
            str(policy).strip().lower() for policy in declared_observed
        )
        if normalized_declared != sorted(observed_policies):
            raise ValueError(
                "Manifest observed_preprocessing_policies does not match trained-cycle summaries."
            )
        unsupported = set(normalized_declared) - SUPPORTED_PREPROCESSING_POLICIES
        if unsupported:
            raise ValueError(
                "Manifest observed_preprocessing_policies contains unsupported policies: "
                f"{sorted(unsupported)}."
            )
    return payload, p.parent


def _resolve_manifest_path(
    raw_path: Optional[str],
    manifest_dir: Path,
    cycle_id: Optional[int] = None,
) -> Optional[str]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return str((manifest_dir / candidate).resolve())

    if candidate.exists():
        return str(candidate.resolve())

    # Backward compatibility for old HPC-generated absolute Linux paths copied to another machine.
    fallbacks = []
    cycle_fragment = None
    for idx, part in enumerate(candidate.parts):
        if str(part).startswith("cycle_"):
            cycle_fragment = Path(*candidate.parts[idx:])
            break
    if cycle_fragment is not None:
        fallbacks.append((manifest_dir / cycle_fragment).resolve())

    if cycle_id is not None:
        fallbacks.append((manifest_dir / f"cycle_{_cycle_key(cycle_id)}" / candidate.name).resolve())

    fallbacks.append((manifest_dir / candidate.name).resolve())
    for fallback in fallbacks:
        if fallback.exists():
            return str(fallback)

    return str(candidate)


def _resolve_manifest_cycle(
    manifest: dict,
    manifest_dir: Path,
    cycle_id: int,
    strict: bool,
    visited: Optional[set] = None,
) -> Optional[dict]:
    visited = visited or set()
    key = _cycle_key(cycle_id)
    if key in visited:
        raise ValueError(f"Cycle alias loop detected while resolving cycle={key}.")
    visited.add(key)

    entry = manifest.get("cycles", {}).get(key)
    if entry is None:
        if strict:
            raise KeyError(f"Cycle {key} not present in manifest.")
        return None

    status = str(entry.get("status", "")).strip().lower()
    if status == "trained":
        model_path = _resolve_manifest_path(entry.get("model_path"), manifest_dir, cycle_id=int(cycle_id))
        meta_path = _resolve_manifest_path(entry.get("meta_path"), manifest_dir, cycle_id=int(cycle_id))
        scaler_path = _resolve_manifest_path(entry.get("scaler_path"), manifest_dir, cycle_id=int(cycle_id))
        if not model_path or not meta_path or not scaler_path:
            if strict:
                raise ValueError(f"Cycle {key} is trained but model_path/meta_path/scaler_path are missing.")
            return None
        model_exists = Path(model_path).exists()
        meta_exists = Path(meta_path).exists()
        scaler_exists = Path(scaler_path).exists()
        if strict and (not model_exists or not meta_exists or not scaler_exists):
            raise FileNotFoundError(
                f"Cycle {key} paths do not exist after resolution: "
                f"model_path={model_path} (exists={model_exists}), "
                f"meta_path={meta_path} (exists={meta_exists}), "
                f"scaler_path={scaler_path} (exists={scaler_exists})"
            )
        if not strict and (not model_exists or not meta_exists or not scaler_exists):
            return None
        manifest_preprocessing = _manifest_preprocessing_summary(entry)
        metadata_preprocessing = _metadata_preprocessing_summary(Path(meta_path))
        for field in ("policy", "version", "contract_hash", "passthrough_count"):
            if manifest_preprocessing[field] != metadata_preprocessing[field]:
                raise ValueError(
                    "Manifest/model metadata preprocessing mismatch for "
                    f"cycle {key}, field={field}: "
                    f"manifest={manifest_preprocessing[field]!r}, "
                    f"metadata={metadata_preprocessing[field]!r}."
                )
        resolved = dict(entry)
        resolved["resolved_cycle_id"] = int(entry.get("cycle_id", int(cycle_id)))
        resolved["model_path"] = model_path
        resolved["meta_path"] = meta_path
        resolved["scaler_path"] = scaler_path
        resolved["preprocessing_policy"] = manifest_preprocessing["policy"]
        resolved["preprocessing_policy_version"] = manifest_preprocessing["version"]
        resolved["preprocessing_contract_hash"] = manifest_preprocessing[
            "contract_hash"
        ]
        resolved["binary_passthrough_feature_count"] = manifest_preprocessing[
            "passthrough_count"
        ]
        return resolved

    if status == "alias":
        alias_to = entry.get("alias_to")
        if alias_to is None:
            if strict:
                raise ValueError(f"Cycle {key} has status=alias but no alias_to.")
            return None
        return _resolve_manifest_cycle(
            manifest=manifest,
            manifest_dir=manifest_dir,
            cycle_id=int(alias_to),
            strict=strict,
            visited=visited,
        )

    if strict:
        raise ValueError(f"Cycle {key} unavailable in manifest (status={status!r}).")
    return None
