#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for the IsolationForest anomaly helper.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
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
    return df[keep]


def plot_raw_timeline(
    df_plot: pd.DataFrame,
    maintenance_windows: List[Tuple],
    save_fig: Optional[str],
    train_frac: float = None,
    train_cutoff_time: Optional[pd.Timestamp] = None,
    min_window_minutes: float = 5.0,
    show_window_labels: bool = True,
    window_label_fontsize: int = 9,
    window_label_format: str = "{id}",
    feature_label: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(14, 5))
    normal = df_plot[df_plot["is_anomaly"] == 0]
    anom = df_plot[df_plot["is_anomaly"] == 1]

    ax.scatter(normal.index, normal["y_plot"], s=8, alpha=0.7, label="Normal")
    ax.scatter(anom.index, anom["y_plot"], s=10, alpha=0.9, label="Anomalous")

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

    # Visualize maintenance windows
    span_drawn = False
    vline_drawn = False
    min_sec = max(0.0, float(min_window_minutes)) * 60.0

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

        dur_sec_real = (e - s).total_seconds()
        dur_min_real = max(0.0, dur_sec_real / 60.0)
        dur_sec = (e_clip - s_clip).total_seconds()

        if dur_sec <= min_sec:
            ax.axvline(s_clip, color="#FFC107", alpha=0.8, linewidth=1.6)
            vline_drawn = True
            x_for_label = s_clip
        else:
            ax.axvspan(s_clip, e_clip, color="#FFC107", alpha=0.25, lw=0)
            span_drawn = True
            x_for_label = s_clip + (e_clip - s_clip) / 2

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
                    x_for_label, lanes_y[lane_idx], label,
                    transform=xaxis_transform,
                    fontsize=window_label_fontsize,
                    color="#5D4037",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FFC107", alpha=0.9, linewidth=0.8)
                )
            except Exception:
                pass

    handles, labels = ax.get_legend_handles_labels()
    if span_drawn:
        handles.append(Patch(facecolor="#FFC107", alpha=0.25, label="Failure window"))
        labels.append("Failure window")
    if vline_drawn:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#FFC107", lw=1.6, alpha=0.8, label="Short failure (line)"))
        labels.append("Short failure (line)")
    ax.legend(handles, labels, loc="best")

    ax.set_xlabel("Time")
    ax.set_ylabel(feature_label)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    try:
        fig.subplots_adjust(top=0.88)
    except Exception:
        pass
    if save_fig:
        fig.savefig(save_fig, dpi=160, bbox_inches="tight")
