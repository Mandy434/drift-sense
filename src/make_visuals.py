#!/usr/bin/env python3
"""
Render a side-by-side (reference | search) visual for one pair: the search
image gets a green box at the true center and a red cross at the localiser's
prediction, so a reader can see the match (or miss) at a glance.

Usage
-----
    python src/make_visuals.py --dataset results/dataset  --idx 0  --out results/examples/dram_success.png
    python src/make_visuals.py --dataset results/sweep202 --idx 12 --out results/examples/failure_case.png

Or let it pick automatically within one dataset:
    python src/make_visuals.py --dataset results/dataset --auto success --out results/examples/success_case.png
    python src/make_visuals.py --dataset results/dataset --auto failure --out results/examples/failure_case.png

`--auto failure` runs every pair in the dataset and renders the worst one --
useful for a dataset that actually contains a failure. The default 30-pair
baseline seeds mostly don't (that's the point of a 93.33% accuracy figure);
the documented worst-case (721 px, seed 202 pair 12 -- see README "Results")
is reproduced with:

    python src/generate_dataset.py --num-pairs 30 --out results/sweep202 --style mixed --seed 202
    python src/make_visuals.py --dataset results/sweep202 --idx 12 --out results/examples/failure_case.png
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve localize.py next to this file, not relative to the caller's CWD --
# see the identical note in evaluate.py.
LOCALIZE = Path(__file__).resolve().parent / "localize.py"


def _predict(dataset, rec):
    out = subprocess.run(
        [sys.executable, str(LOCALIZE),
         "--reference", f"{dataset}/{rec['reference']}",
         "--search", f"{dataset}/{rec['search']}"],
        capture_output=True, text=True).stdout.split()
    return float(out[0]), float(out[1])


def render(dataset, idx, outname):
    rec = json.load(open(f"{dataset}/ground_truth.json"))[idx]
    ref = cv2.imread(f"{dataset}/{rec['reference']}", cv2.IMREAD_UNCHANGED)
    sea = cv2.imread(f"{dataset}/{rec['search']}", cv2.IMREAD_UNCHANGED)
    if ref is None or sea is None:
        raise SystemExit(f"couldn't read pair {idx} from '{dataset}/' -- "
                          f"did you generate it first?")

    cx, cy = _predict(dataset, rec)
    tx, ty = rec["true_center_x"], rec["true_center_y"]
    err = float(np.hypot(cx - tx, cy - ty))

    if sea.ndim == 2:
        sea = cv2.cvtColor(sea, cv2.COLOR_GRAY2BGR)
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    s = 50
    cv2.rectangle(sea, (int(tx - s), int(ty - s)), (int(tx + s), int(ty + s)),
                  (0, 220, 0), 2)
    cv2.drawMarker(sea, (int(round(cx)), int(round(cy))), (0, 0, 255),
                    cv2.MARKER_CROSS, 30, 2)

    ref_disp = cv2.resize(ref, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    cv2.putText(ref_disp, "REFERENCE (100x)", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(sea, f"SEARCH (10x)  green=true  red=predicted  err={err:.2f}px",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    combo = np.hstack([ref_disp, np.full((1000, 20, 3), 40, np.uint8), sea])
    cv2.imwrite(outname, combo)
    print(f"{outname}: pair {idx} ({rec.get('style')}, {rec.get('modality')}), "
          f"err={err:.2f}px  ({'OK' if err <= 5.0 else 'FAIL'} @ 5px)")


def auto_pick(dataset, which):
    recs = json.load(open(f"{dataset}/ground_truth.json"))
    errs = []
    for r in recs:
        cx, cy = _predict(dataset, r)
        errs.append(float(np.hypot(cx - r["true_center_x"], cy - r["true_center_y"])))
    idx = int(np.argmin(errs)) if which == "success" else int(np.argmax(errs))
    print(f"auto-picked pair {idx} ({which}), err={errs[idx]:.2f}px")
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset directory to read the pair from")
    ap.add_argument("--idx", type=int, default=None, help="pair index to render")
    ap.add_argument("--auto", choices=["success", "failure"], default=None,
                     help="instead of --idx, scan the whole dataset and pick "
                          "the best (success) or worst (failure) pair")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    if args.idx is None and args.auto is None:
        raise SystemExit("pass either --idx N or --auto {success,failure}")
    idx = args.idx if args.idx is not None else auto_pick(args.dataset, args.auto)
    render(args.dataset, idx, args.out)


if __name__ == "__main__":
    main()
