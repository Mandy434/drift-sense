#!/usr/bin/env python3
"""
Run localize.py on a dataset dir and score against ground truth.

Reports, per the Applied Materials validation requirements:
  - Pass rate at 5, 4, 2 and 1 pixel thresholds, plus sub-pixel share.
  - Mean, median and worst-case (max) error.
  - Runtime per pair, with hardware, Python version and timing method stated.
  - A CSV manifest with reference path, search path, true (x, y), predicted
    (x, y), error, per-pair runtime and generation metadata.
  - An accuracy-vs-threshold bar chart (results.png) alongside results.csv.

Usage
-----
    python evaluate.py dataset
    python evaluate.py dataset --csv results.csv --plot results.png
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Resolve localize.py next to this file rather than relying on the caller's
# current working directory -- evaluate.py is invoked as `python src/evaluate.py
# results/dataset` from the repo root, so a bare "localize.py" would look for
# ./localize.py relative to the CWD and fail to find it in src/.
LOCALIZE = Path(__file__).resolve().parent / "localize.py"


def main():
    ap = argparse.ArgumentParser(description="Score localize.py against a generated dataset's ground truth.")
    ap.add_argument("dataset", nargs="?", default="demo10",
                     help="dataset directory written by generate_dataset.py")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 4.0, 2.0, 1.0],
                     help="pixel tolerances to report pass rate at (default: 5 4 2 1)")
    ap.add_argument("--csv", default=None,
                     help="write per-pair results to this CSV (default: <dataset>/results.csv)")
    ap.add_argument("--plot", default=None,
                     help="write an accuracy-vs-threshold bar chart here "
                          "(default: <dataset>/results.png; pass --no-plot to skip)")
    ap.add_argument("--no-plot", action="store_true", help="skip writing the chart")
    args = ap.parse_args()

    d = args.dataset
    tol = args.thresholds[0]                       # headline tolerance, kept for the printed summary line
    csv_path = args.csv or f"{d}/results.csv"
    plot_path = None if args.no_plot else (args.plot or f"{d}/results.png")

    recs = json.load(open(f"{d}/ground_truth.json"))

    # Timing method: wall-clock around the localize.py subprocess call, using
    # time.perf_counter() (a monotonic, high-resolution clock -- not affected
    # by system clock adjustments, unlike time.time()). This includes process
    # start-up overhead, which is the honest cost of the CLI contract PS asks
    # for ("process a pair ... without manual source-code changes").
    rows = []
    for r in recs:
        t0 = time.perf_counter()
        out = subprocess.run(
            [sys.executable, str(LOCALIZE),
             "--reference", f"{d}/{r['reference']}",
             "--search", f"{d}/{r['search']}"],
            capture_output=True, text=True).stdout.split()
        dt = time.perf_counter() - t0
        cx, cy = float(out[0]), float(out[1])
        err = float(np.hypot(cx - r["true_center_x"], cy - r["true_center_y"]))
        row = {
            "pair_id": r["pair_id"],
            "run_seed": r.get("run_seed"),
            "pair_seed": r.get("pair_seed"),
            "reference": r["reference"],
            "search": r["search"],
            "style": r.get("style"),
            "modality": r.get("modality"),
            "true_center_x": r["true_center_x"],
            "true_center_y": r["true_center_y"],
            "pred_x": cx,
            "pred_y": cy,
            "error_px": round(err, 4),
            "time_s": round(dt, 4),
            "rotation_deg": r.get("rotation_deg"),
            "scale_jitter": r.get("scale_jitter"),
            "pv_amplitude": r.get("pv_amplitude"),
            "on_mat_boundary": r.get("on_mat_boundary"),
        }
        rows.append(row)
        flag = "OK " if err <= tol else "FAIL"
        print(f"{flag} pair {r['pair_id']:>3}: pred=({cx:7.2f},{cy:7.2f}) "
              f"true=({r['true_center_x']:7.2f},{r['true_center_y']:7.2f}) "
              f"err={err:6.2f}px  {dt:.2f}s")

    errs = np.array([row["error_px"] for row in rows])
    times = np.array([row["time_s"] for row in rows])

    # ---- CSV manifest: predictions + ground truth + per-pair metadata ----
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # ---- threshold-wise pass rate ----
    print(f"\nPass rate by tolerance ({len(errs)} pairs):")
    thresh_results = []
    for t in sorted(args.thresholds, reverse=True):
        n_pass = int((errs <= t).sum())
        pct = 100.0 * n_pass / len(errs)
        thresh_results.append((t, pct, n_pass))
        print(f"  <= {t:g} px : {pct:5.1f}%  ({n_pass}/{len(errs)})")
    sub_pixel = 100.0 * (errs <= 1.0).sum() / len(errs)

    # ---- error statistics ----
    print(f"\nError (px): mean={errs.mean():.3f}  median={np.median(errs):.3f}  "
          f"worst-case={errs.max():.3f}  min={errs.min():.3f}")

    # ---- runtime, with hardware / Python version / timing method stated ----
    print(f"\nRuntime: mean={times.mean():.3f}s  median={np.median(times):.3f}s  "
          f"max={times.max():.3f}s per pair")
    print(f"Hardware: {platform.processor() or platform.machine()} "
          f"({platform.system()} {platform.release()})")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print("Timing method: time.perf_counter() around each localize.py subprocess "
          "call (wall clock, includes process start-up).")

    print(f"\nAccuracy @ {tol:g}px tolerance: {100.0 * (errs <= tol).mean():.0f}% "
          f"({int((errs <= tol).sum())}/{len(errs)})")
    print(f"Median error: {np.median(errs):.2f}px | Mean time/pair: {times.mean():.2f}s")
    print(f"\nWrote per-pair results to {csv_path}")

    # ---- accuracy-vs-threshold plot ----
    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [f"{t:g}px" for t, _, _ in thresh_results]
            pcts = [p for _, p, _ in thresh_results]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(labels, pcts, color="#4c72b0")
            for b, p in zip(bars, pcts):
                ax.text(b.get_x() + b.get_width() / 2, p + 1.5, f"{p:.0f}%",
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Pass rate (%)")
            ax.set_title(f"{d}: accuracy by pixel tolerance ({len(errs)} pairs)")
            ax.axhline(sub_pixel, color="gray", linestyle="--", linewidth=0.8)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Wrote chart to {plot_path}")
        except ImportError:
            print("matplotlib not installed -- skipped the plot (pip install matplotlib, "
                  "or pass --no-plot to silence this).")


if __name__ == "__main__":
    main()
