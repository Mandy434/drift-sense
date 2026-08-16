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
    python localize.py --reference ref.png --search search.png --with-confidence

Output: a single line "x y" on stdout (predicted center, sub-pixel floats).
With --with-confidence, a third number is appended: "x y confidence" -- the
winning site's normalized cross-correlation score, repeatable for a given
input pair. The default two-number output is unchanged unless this flag is
given.

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

NOMINAL_SCALE_RATIO = 10.0     # SEM: reference is 100x, search is 10x
OPTICAL_SCALE_RATIO = 3.0      # optical: a diffraction-limited objective cannot
                               # deliver a 1 um reference field, so the optical
                               # reference/search pair is ~3x, not 10x
SEARCH_BLUR = 1.5              # denoising sigmas (in pixels)
TEMPLATE_BLUR = 1.0
PEAK_MARGIN = 0.001            # combined-score margin for true ambiguity
CANDIDATE_MARGIN = 0.025       # lattice-NCC margin for stage-2 candidates
PITCH_SIGMA = 8.0              # low-pass sigma > array pitch (px)
FP_CONFIDENT = 0.70            # min fp correlation to trust an override
FP_GAP = 0.03                  # min fp advantage over the lattice winner
FP_MAX_DEFICIT = 0.02          # max lattice deficit an override may have
NMS_RADIUS = 12                # px, suppress duplicate detections of one site


def load_image(path):
    """
    Load a capture as float32.

    SEM captures are single-channel electron images; optical-microscope captures
    are 3-channel RGB. Both are returned as-is rather than flattening colour to
    grey, because on an optical image the across-die film-thickness variation
    shows up largely as a HUE shift -- converting to luminance throws away the
    most discriminative part of the fingerprint. Every stage below
    (matchTemplate, Gaussian blur, affine warp, the fingerprint correlation)
    operates on 1- and 3-channel arrays alike, so the pipeline itself is
    modality-agnostic; OpenCV's normalised cross-correlation simply sums over
    channels.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"ERROR: cannot read image '{path}'")
    if img.ndim == 3 and img.shape[2] == 4:            # drop an alpha channel
        img = img[..., :3]
    return img.astype(np.float32)


# kept for backwards compatibility with anything importing the old name
def load_gray(path):
    img = load_image(path)
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def make_template(ref, size, angle):
    """Shrink the reference to `size` px and rotate by `angle` degrees.
    Works for 1- and 3-channel references alike."""
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


def localize(reference, search, verbose=False, scale_ratio=None,
            with_confidence=False):
    t0 = time.time()
    search_f = cv2.GaussianBlur(search, (0, 0), SEARCH_BLUR)

    # The reference/search magnification pair differs between modalities: 10x for
    # SEM, ~3x for optical (an optical objective cannot deliver a 1 um reference
    # field). Rather than being told which, infer the footprint by scanning a
    # wide range of template sizes and letting correlation decide.
    ratio = OPTICAL_SCALE_RATIO if search.ndim == 3 else NOMINAL_SCALE_RATIO
    if scale_ratio is not None:
        ratio = scale_ratio
    nominal = int(round(search.shape[0] / ratio))

    # ---- coarse grid over scale x rotation -------------------------------
    # +/-3 steps of ~4% covers +/-12%, comfortably spanning the 9:1-11:1
    # scale-ratio robustness range the problem statement calls out (a
    # search image generated at scale_ratio=11 sits ~9% below nominal;
    # at scale_ratio=9, ~11% above) without being told the true ratio.
    step = max(2, int(round(nominal * 0.04)))
    sizes = [nominal + d * step for d in (-3, -2, -1, 0, 1, 2, 3)]
    angles = [-2.0, -1.0, 0.0, 1.0, 2.0]
    best = (-2.0, None, None, None)                 # score, size, angle, res
    for s in sizes:
        for a in angles:
            score, res = best_of_config(search_f, reference, s, a)
            if score > best[0]:
                best = (score, s, a, res)

    # ---- fine grid around the coarse optimum -----------------------------
    s0, a0 = best[1], best[2]
    fine = max(1, step // 2)
    for s in range(s0 - 2 * fine, s0 + 2 * fine + 1, fine):
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
        if patch.shape != tpl_lp.shape:                # partial window at border
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

    # Repeatable confidence for the site actually returned: the chosen
    # candidate's own lattice NCC score (TM_CCOEFF_NORMED, nominally in
    # [-1, 1], typically 0.4-1.0 for a correct lock). This is deliberately
    # `chosen["pv"]`, not the raw grid-search `score` above -- the two
    # usually coincide, but on a fingerprint override or a genuine
    # nearest-to-center tie-break the final site differs from the coarse/fine
    # grid's naive best, and `score` would then describe a site that was not
    # the one actually reported.
    confidence = float(chosen["pv"])

    if verbose:
        print(f"# grid_best_score={score:.4f} size={size}px angle={angle:+.2f}deg "
              f"candidates={len(cand)} chosen_confidence={confidence:.4f} "
              f"time={time.time() - t0:.2f}s", file=sys.stderr)
    if with_confidence:
        return cx, cy, confidence
    return cx, cy


def main():
    ap = argparse.ArgumentParser(description="Drift-Sense localization inference")
    ap.add_argument("--reference", required=True, help="path to reference image")
    ap.add_argument("--search", required=True, help="path to search image")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--scale-ratio", type=float, default=None,
                    help="override the reference/search magnification ratio "
                         "(default: 10 for SEM, 3 for 3-channel optical)")
    ap.add_argument("--with-confidence", action="store_true",
                    help="also print a third number: the winning site's "
                         "normalized cross-correlation score (TM_CCOEFF_NORMED, "
                         "roughly [-1, 1], repeatable for a given input pair) as "
                         "a confidence value. Does not change the default "
                         "two-number 'x y' output -- only adds a third number "
                         "when this flag is given.")
    args = ap.parse_args()

    ref = load_image(args.reference)
    sea = load_image(args.search)
    if (ref.ndim == 3) != (sea.ndim == 3):
        sys.exit("ERROR: reference and search must be the same modality "
                 "(both 1-channel SEM or both 3-channel optical)")
    if args.with_confidence:
        cx, cy, conf = localize(ref, sea, verbose=args.verbose,
                                scale_ratio=args.scale_ratio,
                                with_confidence=True)
        print(f"{cx:.2f} {cy:.2f} {conf:.4f}")
    else:
        cx, cy = localize(ref, sea, verbose=args.verbose,
                          scale_ratio=args.scale_ratio)
        print(f"{cx:.2f} {cy:.2f}")


if __name__ == "__main__":
    main()
