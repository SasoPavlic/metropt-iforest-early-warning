#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modeling utilities for the IsolationForest anomaly helper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def lpf_exponential(x: pd.Series, alpha: float) -> pd.Series:
    """Simple exponential smoothing y[i]=y[i-1]+alpha*(x[i]-y[i-1])."""
    if alpha is None or alpha <= 0:
        return x
    y = np.empty_like(x.values, dtype=float)
    if len(x) == 0:
        return x
    y[0] = x.iloc[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x.iloc[i] - y[i - 1])
    return pd.Series(y, index=x.index, name=f"{x.name}_lpf")


def train_iforest_and_score(
    X: pd.DataFrame,
    train_frac: float = 0.10,
    lpf_alpha: float = 0.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, dict]:
    n = len(X)
    n_train = max(30, int(n * train_frac))
    X_train = X.iloc[:n_train]
    X_all = X

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    model = IsolationForest(
        n_estimators=100, contamination='auto', random_state=random_state, n_jobs=-1
    )
    model.fit(X_train_s)

    # Larger score => more anomalous
    scores_all = -model.decision_function(X_all_s)
    scores_all = pd.Series(scores_all, index=X.index, name="anom_score")

    # Optional LPF smoothing
    if lpf_alpha and lpf_alpha > 0:
        scores_all_sm = lpf_exponential(scores_all, lpf_alpha)
        train_scores = scores_all_sm.iloc[:n_train]
        score_used = scores_all_sm.rename("anom_score_sm")
    else:
        train_scores = scores_all.iloc[:n_train]
        score_used = scores_all

    # Thresholding via Q3 + 3*IQR on the training portion
    train_vals = train_scores.values.astype(float)
    q1, q3 = np.nanpercentile(train_vals, [25, 75])
    if np.isnan(q1) or np.isnan(q3):
        finite = train_vals[np.isfinite(train_vals)]
        fallback = float(finite.mean()) if finite.size else 0.0
        q1 = q3 = fallback
    iqr = max(0.0, float(q3 - q1))
    thr = float(q3 + 3.0 * iqr)
    if not np.isfinite(thr):
        finite = train_vals[np.isfinite(train_vals)]
        thr = float(finite.max()) if finite.size else float(0.0)
    is_anom = np.where(score_used.values > thr, 1, 0)
    rule = f"Q3+3*IQR (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})"

    out = pd.DataFrame(index=X.index)
    out["anom_score"] = scores_all.values
    if lpf_alpha and lpf_alpha > 0:
        out["anom_score_lpf"] = score_used.values
    out["is_anomaly"] = is_anom

    info = {
        "n_total": int(n),
        "n_train": int(n_train),
        "pct_anom": float(out["is_anomaly"].mean()),
        "threshold": thr,
        "label_rule": rule,
        "pca_components": None,
        "n_features": int(X.shape[1]),
        "lpf_alpha": float(lpf_alpha) if lpf_alpha else 0.0,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
    }
    return out, info
