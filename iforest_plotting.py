#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for the IsolationForest anomaly helper.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def prepare_plot_frame(
    raw_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_to_plot: Optional[str],
    plot_stride: int,
    plot_rolling: Optional[str]
) -> pd.DataFrame:
    # Avoid double-join if predictions already merged
    if {"is_anomaly", "anom_score"}.issubset(set(raw_df.columns)):
        df = raw_df.copy()
    else:
        pred_join = pred_df
        overlap = [c for c in pred_join.columns if c in raw_df.columns]
        if overlap:
            pred_join = pred_join.drop(columns=overlap)
        df = raw_df.join(pred_join, how="inner")

    if feature_to_plot and feature_to_plot in raw_df.columns:
        df["y_plot"] = raw_df[feature_to_plot]
    else:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric column available for plotting.")
        df["y_plot"] = raw_df[numeric_cols[0]]

    if plot_rolling:
        df["y_plot"] = df["y_plot"].rolling(plot_rolling, min_periods=1).median()

    if plot_stride and plot_stride > 1:
        df = df.iloc[::plot_stride].copy()

    keep = ["y_plot", "is_anomaly"]
    if "anom_score_lpf" in df.columns:
        keep.append("anom_score_lpf")
    if "operation_phase" in df.columns:
        keep.append("operation_phase")
    if "maintenance_risk" in df.columns:
        keep.append("maintenance_risk")
    return df[keep]


def plot_raw_timeline(
    df_plot: pd.DataFrame,
    maintenance_windows: List[Tuple],
    save_fig: Optional[str],
    train_frac: float = None,
    train_cutoff_time: Optional[pd.Timestamp] = None,
    show_window_labels: bool = True,
    window_label_fontsize: int = 9,
    window_label_format: str = "{id}",
    risk_alarm_mask: Optional[pd.Series] = None,
    risk_threshold: Optional[float] = None,
    early_warning_minutes: float = 120.0,
):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    if "maintenance_risk" not in df_plot.columns:
        raise ValueError("maintenance_risk column required for plotting.")

    state = (
        risk_alarm_mask.reindex(df_plot.index).fillna(0).astype(bool)
        if risk_alarm_mask is not None
        else pd.Series(False, index=df_plot.index)
    )

    ax.fill_between(
        df_plot.index,
        0,
        1,
        where=~state.values,
        color="#2E7D32",
        alpha=0.2,
        step="post",
        label="Normal",
    )
    risk_label = "Risk alarm"
    if risk_threshold is not None:
        risk_label = f"Risk alarm (≥ θ={risk_threshold:.2f})"
    ax.fill_between(
        df_plot.index,
        0,
        1,
        where=state.values,
        color="#C62828",
        alpha=0.55,
        step="post",
        label=risk_label,
    )

    # Training cutoff
    cutoff_ts = train_cutoff_time
    if cutoff_ts is None and train_frac is not None:
        train_size = int(len(df_plot) * train_frac)
        if 0 < train_size < len(df_plot):
            cutoff_ts = df_plot.index[train_size - 1]
    if cutoff_ts is not None:
        ax.axvline(cutoff_ts, color="#7B1FA2", linestyle="-", linewidth=2.0, alpha=0.9, label="Training cutoff")

    # Determine plot x-range to clip maintenance windows
    if len(df_plot.index) > 0:
        xmin = pd.to_datetime(df_plot.index.min())
        xmax = pd.to_datetime(df_plot.index.max())
    else:
        xmin = xmax = None

    # Visualize maintenance windows + early-warning lead
    span_drawn = False
    warn_drawn = False
    horizon = pd.to_timedelta(float(max(0.0, early_warning_minutes)), unit="m")

    # Label placement lanes
    lanes_y = [1.02, 1.06, 1.10]
    last_x_in_lane = [float('-inf')] * len(lanes_y)
    xaxis_transform = ax.get_xaxis_transform()
    xnum_min = mdates.date2num(xmin) if xmin is not None else None
    xnum_max = mdates.date2num(xmax) if xmax is not None else None
    xspan_num = (xnum_max - xnum_min) if (xnum_min is not None and xnum_max is not None) else None
    sep_thresh = (xspan_num / 40.0) if xspan_num else 0.0

    for item in maintenance_windows or []:
        if len(item) >= 4:
            s_raw, e_raw, wid, sev = item[0], item[1], item[2], item[3]
        else:
            s_raw, e_raw = item[0], item[1]
            wid, sev = None, None
        s = pd.to_datetime(s_raw)
        e = pd.to_datetime(e_raw)
        if xmin is not None and xmax is not None:
            if e < xmin or s > xmax:
                continue
            s_clip = max(s, xmin)
            e_clip = min(e, xmax)
        else:
            s_clip, e_clip = s, e

        dur_min_real = max(0.0, (e - s).total_seconds() / 60.0)

        ax.axvspan(s_clip, e_clip, color="#FFC107", alpha=0.25, lw=0)
        span_drawn = True
        x_for_label = s_clip + (e_clip - s_clip) / 2

        if horizon > pd.Timedelta(0):
            warn_start = s - horizon
            warn_start = max(warn_start, xmin) if xmin is not None else warn_start
            ax.axvspan(warn_start, s, color="#90CAF9", alpha=0.18, lw=0)
            ax.axvline(warn_start, color="#0288D1", linestyle="--", linewidth=1.2)
            ax.axvline(s, color="#0288D1", linestyle="--", linewidth=1.2)
            warn_drawn = True

        if show_window_labels:
            try:
                label_id = (wid if wid is not None else "")
                label_sev = (sev if sev is not None else "")
                label = window_label_format.format(id=label_id, severity=label_sev, dur_min=int(round(dur_min_real)))
                if str(label).strip() == "":
                    continue
                xnum = mdates.date2num(pd.to_datetime(x_for_label))
                lane_idx = 0
                if xspan_num:
                    for j in range(len(lanes_y)):
                        idx = j % len(lanes_y)
                        if xnum - last_x_in_lane[idx] >= sep_thresh:
                            lane_idx = idx
                            break
                        lane_idx = idx
                    last_x_in_lane[lane_idx] = xnum
                ax.text(
                    x_for_label,
                    lanes_y[lane_idx],
                    label,
                    transform=xaxis_transform,
                    fontsize=window_label_fontsize,
                    color="#5D4037",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FFC107", alpha=0.9, linewidth=0.8),
                )
            except Exception:
                pass

    handles, labels = ax.get_legend_handles_labels()
    if span_drawn:
        handles.append(Patch(facecolor="#FFC107", alpha=0.25, label="Failure window"))
        labels.append("Failure window")
    if warn_drawn:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#0288D1", linestyle="--", lw=1.2, label="Early-warning horizon"))
        labels.append("Early-warning horizon")
        handles.append(Patch(facecolor="#90CAF9", alpha=0.18, label="Lead window"))
        labels.append("Lead window")
    ax.legend(handles, labels, loc="best")

    ax.set_xlabel("Time")
    ax.set_ylabel("Risk state")
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Normal", "Risk"])
    ax.set_ylim(0, 1)
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=160, bbox_inches="tight")
    plt.show()
