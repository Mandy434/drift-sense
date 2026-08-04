#!/usr/bin/env python3
"""
Drift-Sense Localization Inference Script
=========================================
Finds where the (100x magnification) Reference pattern appears inside the
(10x magnification) Search image and prints the center (x, y) in search-image
pixel coordinates.

Usage
-----
    python localize.py --reference path/to/reference.png --search path/to/search.png

Output: a single line "x y" on stdout (predicted center, sub-pixel floats).

Method (classical, no training required)
----------------------------------------
1. DENOISE      Both captures carry independent sensor noise; a mild Gaussian
                blur suppresses it while preserving the layout structure.
2. COARSE GRID  The nominal magnification ratio is 10x but the true scale and
                rotation are slightly off (stage/optics error). We grid-search
                template size (scale) and rotation, scoring each with
                normalized cross-correlation (NCC, TM_CCOEFF_NORMED).
3. FINE GRID    Refine scale/rotation around the coarse optimum.
4. PERIODICITY  In a periodic array many peaks are near-identical. We extract
   HANDLING     all peaks within a margin of the best score (non-max
                suppression) and, per the challenge rule, choose the candidate
                CLOSEST TO THE CENTER of the search image.
5. SUB-PIXEL    Quadratic (parabolic) fit on the NCC surface around the chosen
                peak gives sub-pixel localization.
"""

import argparse
import sys
import time

import cv2
import numpy as np

NOMINAL_SCALE_RATIO = 10.0     # reference is 100x, search is 10x
SEARCH_BLUR = 1.5              # denoising sigmas (in pixels)
TEMPLATE_BLUR = 1.0
PEAK_MARGIN = 0.001            # combined-score margin for true ambiguity
CANDIDATE_MARGIN = 0.025       # lattice-NCC margin for stage-2 candidates
PITCH_SIGMA = 8.0              # low-pass sigma > array pitch (px)
FP_CONFIDENT = 0.70            # min fp correlation to trust an override
FP_GAP = 0.03                  # min fp advantage over the lattice winner
FP_MAX_DEFICIT = 0.02          # max lattice deficit an override may have
NMS_RADIUS = 12                # px, suppress duplicate detections of one site


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERROR: cannot read image '{path}'")
    return img.astype(np.float32)


def make_template(ref, size, angle):
    """Shrink the reference to `size` px and rotate by `angle` degrees."""
    tpl = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    if angle != 0.0:
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        tpl = cv2.warpAffine(tpl, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(tpl, (0, 0), TEMPLATE_BLUR)


def best_of_config(search_f, ref, size, angle):
    tpl = make_template(ref, size, angle)
    res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(res)
    return mx, res


def top_peaks(res, margin, nms_radius):
    """All local maxima within `margin` of the global max (greedy NMS)."""
    r = res.copy()
    _, best, _, _ = cv2.minMaxLoc(r)
    peaks = []
    while True:
        _, val, _, loc = cv2.minMaxLoc(r)
        if val < best - margin or len(peaks) >= 20:
            break
        peaks.append((loc[0], loc[1], val))
        x0, y0 = loc
        r[max(0, y0 - nms_radius):y0 + nms_radius + 1,
          max(0, x0 - nms_radius):x0 + nms_radius + 1] = -1.0
    return peaks


def subpixel(res, x, y):
    """Parabolic fit of the NCC surface around integer peak (x, y)."""
    h, w = res.shape
    if 0 < x < w - 1:
        d = (res[y, x - 1] - res[y, x + 1]) / (
            2 * (res[y, x - 1] - 2 * res[y, x] + res[y, x + 1]) + 1e-12)
        x = x + float(np.clip(d, -0.5, 0.5))
    if 0 < y < h - 1:
        d = (res[y - 1, x if isinstance(x, int) else int(round(x))] -
             res[y + 1, int(round(x))]) / (
            2 * (res[y - 1, int(round(x))] - 2 * res[int(round(y)), int(round(x))]
                 + res[y + 1, int(round(x))]) + 1e-12)
        y = y + float(np.clip(d, -0.5, 0.5))
    return x, y


def localize(reference, search, verbose=False):
    t0 = time.time()
    search_f = cv2.GaussianBlur(search, (0, 0), SEARCH_BLUR)

    nominal = int(round(search.shape[0] / NOMINAL_SCALE_RATIO))  # ~100 px

    # ---- coarse grid over scale x rotation -------------------------------
    sizes = [nominal + d for d in (-8, -4, 0, 4, 8)]
    angles = [-2.0, -1.0, 0.0, 1.0, 2.0]
    best = (-2.0, None, None, None)                 # score, size, angle, res
    for s in sizes:
        for a in angles:
            score, res = best_of_config(search_f, reference, s, a)
            if score > best[0]:
                best = (score, s, a, res)

    # ---- fine grid around the coarse optimum -----------------------------
    s0, a0 = best[1], best[2]
    for s in range(s0 - 2, s0 + 3):
        for a in np.arange(a0 - 0.75, a0 + 0.76, 0.25):
            if s == s0 and abs(a - a0) < 1e-9:
                continue
            score, res = best_of_config(search_f, reference, s, float(a))
            if score > best[0]:
                best = (score, s, float(a), res)

    score, size, angle, res = best

    # ---- stage 2: periodicity-suppressed verification ---------------------
    # The lattice dominates the NCC, so under heavy noise a wrong period can
    # outscore the true site. The discriminative "fingerprint" (per-cell CD /
    # via-size variation, defects) is a LOW-FREQUENCY brightness modulation:
    # low-pass filtering below the array pitch removes the periodic lattice
    # and leaves only the fingerprint. We re-score every candidate peak with
    # this fingerprint NCC and combine the two scores.
    cand = top_peaks(res, CANDIDATE_MARGIN, NMS_RADIUS)
    sea_lp = cv2.GaussianBlur(search, (0, 0), PITCH_SIGMA)
    tpl_lp = cv2.GaussianBlur(make_template(reference, size, angle),
                              (0, 0), PITCH_SIGMA)
    t = tpl_lp - tpl_lp.mean()
    tn = t / (np.linalg.norm(t) + 1e-9)

    scored = []
    for px, py, pv in cand:
        patch = sea_lp[py:py + size, px:px + size]
        if patch.shape != tpl_lp.shape:
            continue
        p = patch - patch.mean()
        fp = float((p / (np.linalg.norm(p) + 1e-9) * tn).sum())
        scored.append({"x": px, "y": py, "pv": pv, "fp": fp})
    if not scored:                                  # degenerate fallback
        scored = [{"x": p[0], "y": p[1], "pv": p[2], "fp": 0.0} for p in cand]

    # Default winner: the lattice-NCC maximum (correct in the vast majority
    # of cases). The fingerprint may OVERRIDE it only on strong evidence:
    #   (a) the fp winner's fingerprint correlation is itself high (a real
    #       "lock", not low-frequency noise),
    #   (b) it beats the lattice winner's fingerprint by a wide gap, and
    #   (c) its lattice deficit is small (it was a plausible match anyway).
    # This rescues period-jump failures without demoting confident wins.
    lat_w = max(scored, key=lambda s: s["pv"])
    fp_w = max(scored, key=lambda s: s["fp"])
    winner = lat_w
    if (fp_w is not lat_w
            and fp_w["fp"] >= FP_CONFIDENT
            and fp_w["fp"] - lat_w["fp"] >= FP_GAP
            and lat_w["pv"] - fp_w["pv"] <= FP_MAX_DEFICIT):
        winner = fp_w

    # ---- residual ambiguity: nearest-to-center rule ------------------------
    # Candidates statistically indistinguishable from the winner (both scores
    # within noise) -> per the challenge rule, take the one closest to the
    # center of the search image.
    ambiguous = [s for s in scored
                 if abs(s["pv"] - winner["pv"]) <= PEAK_MARGIN
                 and abs(s["fp"] - winner["fp"]) <= 0.05]
    ctr = (search.shape[1] / 2.0, search.shape[0] / 2.0)
    chosen = min(
        ambiguous,
        key=lambda s: np.hypot(s["x"] + size / 2 - ctr[0],
                               s["y"] + size / 2 - ctr[1]))
    px, py = chosen["x"], chosen["y"]

    # ---- sub-pixel refinement ---------------------------------------------
    sx, sy = subpixel(res, px, py)
    cx, cy = sx + size / 2.0, sy + size / 2.0

    if verbose:
        print(f"# score={score:.4f} size={size}px angle={angle:+.2f}deg "
              f"candidates={len(cand)} time={time.time() - t0:.2f}s",
              file=sys.stderr)
    return cx, cy


def main():
    ap = argparse.ArgumentParser(description="Drift-Sense localization inference")
    ap.add_argument("--reference", required=True, help="path to reference image")
    ap.add_argument("--search", required=True, help="path to search image")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref = load_gray(args.reference)
    sea = load_gray(args.search)
    cx, cy = localize(ref, sea, verbose=args.verbose)
    print(f"{cx:.2f} {cy:.2f}")


if __name__ == "__main__":
    main()
