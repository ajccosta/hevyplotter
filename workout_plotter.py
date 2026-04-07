#!/usr/bin/env python3
"""
workout_time_series_plots.py (with trend lines)

Usage:
    python workout_time_series_plots.py path/to/export.csv

Output:
    ./plots_by_exercise/<safe_exercise_name>/<metric>.png
Metrics produced per exercise:
    - best_1rm.png          (daily maximum estimated 1RM using Epley)
    - heaviest_weight.png   (daily maximum weight used)
    - total_volume.png      (daily sum of weight * reps)

This version adds a simple linear trend line (least-squares) to each time-series plot.
If there are >= 2 data points the script fits a straight line (numpy.polyfit) on the
matplotlib date numbers; if there is exactly 1 data point it draws a horizontal dashed
line at that value.

Notes:
 - The script expects a CSV-like export with fields similar to your sample.
 - It is robust to missing columns and quoted fields.
"""

from pathlib import Path
import argparse
import csv
import math
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def parse_csv_file(path, date_col_index=1, exercise_col_index=4, weight_col_index=9, reps_col_index=10):
    """Read a CSV-like file without header, return DataFrame with parsed fields."""
    # read with csv.reader to handle quoted fields and uneven rows
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["date","exercise","weight","reps","raw_date"])

    # detect max columns and pad rows so DataFrame creation is safe
    maxcols = max(len(r) for r in rows)
    padded = [r + [""] * (maxcols - len(r)) for r in rows]
    df = pd.DataFrame(padded)

    # safe column selection: if index out of range fill with ""
    def col_or_empty(df, idx):
        return df[idx] if idx in df.columns else pd.Series([""]*len(df))

    df_parsed = pd.DataFrame({
        "raw_date": col_or_empty(df, date_col_index).astype(str).str.strip(),
        "exercise": col_or_empty(df, exercise_col_index).astype(str).str.strip(),
        "weight": col_or_empty(df, weight_col_index),
        "reps": col_or_empty(df, reps_col_index),
    })

    # convert numeric (coerce to NaN)
    df_parsed["weight"] = pd.to_numeric(df_parsed["weight"].replace("", pd.NA), errors="coerce")
    df_parsed["reps"] = pd.to_numeric(df_parsed["reps"].replace("", pd.NA), errors="coerce")

    # parse date -> date only (day resolution). Try multiple common formats.
    def parse_date(s):
        s = (s or "").strip()
        if not s:
            return pd.NaT
        # many exports have "24 Aug 2024, 16:22" or ISO. Try several formats:
        formats = ["%d %b %Y, %H:%M", "%d %B %Y, %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date()
            except Exception:
                pass
        # try to extract date-like substring
        m = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", s)
        if m:
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    return datetime.strptime(m.group(1), fmt).date()
                except Exception:
                    pass
        # fallback: try pandas flexible parser
        try:
            ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            return ts.date()
        except Exception:
            return pd.NaT

    df_parsed["date"] = df_parsed["raw_date"].apply(parse_date)

    # normalize exercise names: empty -> "(unknown)"
    df_parsed["exercise"] = df_parsed["exercise"].replace("", "(unknown)").str.strip()

    return df_parsed


def epley_one_rm(weight, reps):
    """Epley formula: 1RM = w * (1 + reps / 30). Returns NaN if invalid."""
    if weight is None or reps is None:
        return float("nan")
    if math.isnan(weight) or math.isnan(reps):
        return float("nan")
    if reps <= 0:
        return float("nan")
    return float(weight) * (1.0 + float(reps) / 30.0)


def aggregate_metrics(df):
    """Given parsed df with columns date, exercise, weight, reps, compute per-exercise per-date metrics.
    Returns a DataFrame with index [exercise, date] and columns best_1rm, heaviest_weight, total_volume."""
    # compute per-row one_rm and volume
    df = df.copy()
    df["one_rm"] = df.apply(lambda r: epley_one_rm(r["weight"], r["reps"]), axis=1)
    # volume: if weight present and reps present compute weight*reps, else 0
    def volume_of_row(w, r):
        if pd.isna(w) or pd.isna(r):
            return 0.0
        return float(w) * float(r)
    df["volume"] = df.apply(lambda r: volume_of_row(r["weight"], r["reps"]), axis=1)

    # drop rows with no date or exercise
    df = df[df["date"].notna() & df["exercise"].notna()]

    # group
    grouped = df.groupby(["exercise", "date"]).agg(
        best_1rm=("one_rm", "max"),
        heaviest_weight=("weight", "max"),
        total_volume=("volume", "sum"),
        sets_count=("weight", "count"),
    ).reset_index()

    # replace NaN best_1rm/ heaviest with NaN (already) and keep them
    return grouped


def safe_name(s):
    return "".join(c if (c.isalnum() or c in " _-()") else "_" for c in s).strip()[:120]


def plot_time_series(grouped_df, output_dir, exercise, metric_col, metric_label, trend_degree):
    """Plot a single metric for a single exercise across dates and save PNG.
    grouped_df: DataFrame filtered to the exercise with columns date and metric_col."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    df = grouped_df.copy()
    df = df.sort_values("date")
    dates = pd.to_datetime(df["date"])
    values = df[metric_col].astype(float)

    out_path = p / f"{metric_col}.png"

    plt.figure(figsize=(8,4.5))
    ax = plt.gca()

    # If there's no numeric data (all NaN or zeros), show a message
    if values.isna().all() or (values.fillna(0.0) == 0.0).all():
        ax.text(0.5, 0.5, "No data for this metric on any date", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{exercise} — {metric_label}")
        ax.axis("off")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    # Plot line + markers. Do not explicitly set colors (use matplotlib defaults).
    ax.plot(dates, values, marker="o", linestyle="-")
    # annotate each point
    for x, y in zip(dates, values):
        try:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center")
        except Exception:
            pass

    # Add polynomial trend line (fit on matplotlib date numbers)
    x_all = mdates.date2num(dates.dt.to_pydatetime())
    y_all = values.to_numpy(dtype=float)
    valid_mask = ~np.isnan(y_all)
    x_valid = x_all[valid_mask]
    y_valid = y_all[valid_mask]

    n_points = len(x_valid)
    if n_points >= 2:
        # choose the degree we can actually fit: at most n_points-1
        fit_deg = int(min(trend_degree, max(1, n_points - 1)))
        try:
            coeff = np.polyfit(x_valid, y_valid, fit_deg)
            y_fit = np.polyval(coeff, x_all)
            # plot trend as dashed line; if fit_deg>1, label it with degree
            if fit_deg == 1:
                ax.plot(dates, y_fit, linestyle="--", linewidth=1)
            else:
                ax.plot(dates, y_fit, linestyle="--", linewidth=1, label=f"poly deg={fit_deg}")
                ax.legend()
        except Exception:
            # fallback: if fit fails, try linear fit
            try:
                coeff = np.polyfit(x_valid, y_valid, 1)
                y_fit = np.polyval(coeff, x_all)
                ax.plot(dates, y_fit, linestyle="--", linewidth=1)
            except Exception:
                pass
    elif n_points == 1:
        # single data point: draw a horizontal dashed line at that value
        ax.axhline(y=float(y_valid[0]), linestyle="--", linewidth=1)

    ax.set_title(f"{exercise} — {metric_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    # format x-axis for dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def produce_plots(input_csv, output_dir="plots_by_exercise", date_col_index=1, exercise_col_index=4, weight_col_index=9, reps_col_index=10, trend_degree=2):
    df = parse_csv_file(input_csv, date_col_index, exercise_col_index, weight_col_index, reps_col_index)
    agg = aggregate_metrics(df)
    if agg.empty:
        print("No aggregated data found. Exiting.")
        return

    # For each exercise, create a subfolder and produce three pngs (one per metric).
    metrics = [
        ("best_1rm", "Best 1RM (kg)"),
        ("heaviest_weight", "Heaviest weight (kg)"),
        ("total_volume", "Total volume (kg·reps)"),
    ]
    created_files = []
    for exercise, g in agg.groupby("exercise"):
        safe_ex = safe_name(exercise) or "unknown_exercise"
        ex_dir = Path(output_dir) / safe_ex
        ex_dir.mkdir(parents=True, exist_ok=True)
        # ensure we have continuous index by date sorted
        g = g.sort_values("date").reset_index(drop=True)
        for col, label in metrics:
            # if column doesn't exist fill with NaN
            if col not in g.columns:
                g[col] = float("nan")
            out_path = plot_time_series(g[["date", col]], ex_dir, exercise, col, label, trend_degree)
            created_files.append(str(out_path))
    print(f"Created {len(created_files)} plot files in '{output_dir}'. Example file: {created_files[0] if created_files else 'none'}")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Produce time-series plots per exercise/metric from workout export.")
    parser.add_argument("input_csv", help="Path to CSV-like workout export")
    parser.add_argument("--output-dir", default="plots_by_exercise", help="Directory to save plots")
    parser.add_argument("--date-col", type=int, default=1, help="zero-based column index of the date/start column (default 1)")
    parser.add_argument("--exercise-col", type=int, default=4, help="zero-based column index of exercise name (default 4)")
    parser.add_argument("--weight-col", type=int, default=9, help="zero-based column index of weight (default 9)")
    parser.add_argument("--reps-col", type=int, default=10, help="zero-based column index of reps (default 10)")
    parser.add_argument("--trend-degree", type=int, default=2, help="polynomial degree of trend line (default 2)")
    args = parser.parse_args()

    produce_plots(args.input_csv, args.output_dir, args.date_col, args.exercise_col, args.weight_col, args.reps_col, args.trend_degree)


if __name__ == "__main__":
    main()

