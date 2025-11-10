#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics helpers for the IsolationForest anomaly helper.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def window_mask(index: pd.DatetimeIndex, windows: List[Tuple]) -> pd.Series:
    """Return a binary Series (1 inside any failure window, else 0)."""
    if index.size == 0:
        return pd.Series([], dtype=int, index=index)
    mask = np.zeros(index.shape[0], dtype=bool)
    for w in windows or []:
        s = pd.to_datetime(w[0])
        e = pd.to_datetime(w[1])
        mask |= (index >= s) & (index <= e)
    return pd.Series(mask.astype(int), index=index, name="is_failure")


def confusion_and_scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute TP, FP, FN, TN, precision, recall, f1, accuracy."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }
