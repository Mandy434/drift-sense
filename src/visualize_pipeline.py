#!/usr/bin/env python3
"""
Drift-Sense pipeline visualiser
===============================
Renders the generator's imaging chain one stage at a time, so each step in
generate_dataset.py can be seen (and defended in a review) instead of only the
final PNG.

Two figures are produced.

  <out>_stages.png    the SEM chain, panel by panel:
                        1 layout          raw pattern (substrate / lines / vias)
                        2 pv_field        process-variation fingerprint field
                        3 layout x pv     what both captures actually image
                        4 edge_brighten   SE yield rise at topography edges
                        5 beam_blur       Gaussian probe PSF
                        6 shot_noise      Poisson, dose-limited
                        7 read+drift      Gaussian read noise, gain/offset
                        8 structures      pattern components by intensity band

  <out>_pair.png      the same site as a reference / search pair, with the
                      reference footprint outlined in the search image, i.e.
                      how much of the search image one reference actually is.

Only numpy and cv2 are used, so it adds no dependency beyond what
generate_dataset.py already needs.

Usage
-----
    python visualize_pipeline.py --style dram --seed 42 --out figs/dram
    python visualize_pipeline.py --style finfet --seed 7 --out figs/finfet
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# generate_dataset.py lives at the repo root (one level up from this file in
# src/) -- add it to sys.path so the bare import below resolves regardless of
# the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_dataset import (LAYOUTS, generate_pair, process_variation_field)

PANEL = 420
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_u8(img, lo=None, hi=None):
    """Scale a float field to 8-bit for display (never used for the dataset)."""
    f = img.astype(np.float32)
    lo = float(np.min(f)) if lo is None else lo
    hi = float(np.max(f)) if hi is None else hi
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def panel(img, title, sub=""):
    """One labelled tile."""
    g = cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    if g.ndim == 2:
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bar = np.full((46, PANEL, 3), 24, np.uint8)
    cv2.putText(bar, title, (10, 20), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 38), FONT, 0.40, (150, 205, 255), 1, cv2.LINE_AA)
    tile = np.vstack([bar, g])
    return cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(24, 24, 24))


def grid(tiles, cols=4):
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 24))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def stage_figure(style, seed, crop_px, layout_size=8000):
    """Re-run the chain from generate_dataset.py, keeping every intermediate."""
    rng = np.random.default_rng(seed)
    layout = LAYOUTS[style](rng, size=layout_size)
    pv_amp = 0.25
    pv = process_variation_field(rng, layout_size, amplitude=pv_amp)
    modulated = np.clip(layout * pv, 0, 1.4).astype(np.float32)

    # a crop the reference actually sees, so the fine pattern stays visible
    c = layout_size // 2
    h = crop_px // 2
    sl = (slice(c - h, c + h), slice(c - h, c + h))
    lay_c, pv_c, mod_c = layout[sl], pv[sl], modulated[sl]

    # --- chain, mirroring sem_capture() step for step ---------------------
    blur_sigma, dose, read_sigma, edge_gain = 1.0, 220.0, 0.015, 0.32
    gx = cv2.Sobel(mod_c, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mod_c, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (0, 0), 1.5)
    if edges.max() > 0:
        edges /= edges.max()
    s_edge = np.clip(mod_c + edge_gain * edges, 0, 1.4)
    s_blur = cv2.GaussianBlur(s_edge, (0, 0), blur_sigma)
    s_shot = rng.poisson(np.clip(s_blur, 0, None) * dose).astype(np.float32) / dose
    s_read = s_shot + rng.normal(0, read_sigma, s_shot.shape).astype(np.float32)
    s_final = np.clip(s_read * 1.03 - 0.02, 0, 1)

    # --- structural breakdown by intensity band ---------------------------
    # the generator writes substrate low, lines mid, vias/crossings high, so the
    # bands recover the pattern components without changing generate_dataset.py
    comp = np.zeros((*lay_c.shape, 3), np.uint8)
    comp[..., 0] = to_u8((lay_c > 0.05) & (lay_c < 0.40))     # substrate  (blue)
    comp[..., 1] = to_u8((lay_c >= 0.40) & (lay_c < 0.80))    # lines/fins (green)
    comp[..., 2] = to_u8(lay_c >= 0.80)                       # vias       (red)

    tiles = [
        panel(to_u8(lay_c, 0, 1.2), "1  layout",
              f"{style}, {crop_px}px crop of {layout_size}px die"),
        panel(to_u8(pv_c), "2  process-variation field",
              f"amplitude {pv_amp}, correlation ~ die/20"),
        panel(to_u8(mod_c, 0, 1.2), "3  layout x pv field",
              "identical for both captures"),
        panel(to_u8(s_edge, 0, 1.4), "4  edge brightening",
              f"edge_gain {edge_gain}"),
        panel(to_u8(s_blur, 0, 1.4), "5  beam blur (probe PSF)",
              f"sigma {blur_sigma} px"),
        panel(to_u8(s_shot, 0, 1.4), "6  Poisson shot noise",
              f"dose {dose:.0f} e- at signal 1.0"),
        panel(to_u8(s_final, 0, 1.0), "7  read noise + gain/offset",
              f"read sigma {read_sigma}  -> final reference"),
        panel(comp, "8  pattern components",
              "B substrate / G lines / R vias"),
    ]
    return grid(tiles, cols=4)


def pair_figure(style, seed):
    """Reference and search side by side, with the reference footprint marked."""
    rng = np.random.default_rng(seed)
    ref, search, gt = generate_pair(rng, style=style)
    span = gt["ref_span_in_search_px"]
    x, y = gt["true_center_x"], gt["true_center_y"]

    vis = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    h = int(round(span / 2))
    cv2.rectangle(vis, (int(x) - h, int(y) - h), (int(x) + h, int(y) + h),
                  (0, 0, 255), 2)
    cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)

    left = panel(ref, "REFERENCE  1000x1000 @ ~1 nm/px",
                 f"{style}, high dose / sharp probe")
    right = panel(vis, "SEARCH  1000x1000 @ ~10 nm/px",
                  f"true centre ({x}, {y}), footprint {span:.0f}px, "
                  f"rot {gt['rotation_deg']:+.2f} deg")
    fig = np.hstack([left, right])
    note = np.full((40, fig.shape[1], 3), 24, np.uint8)
    cv2.putText(note, f"the reference is {span:.0f}x{span:.0f} px of the search "
                      f"image -- {100 * span * span / (1000 * 1000):.1f}% of its area",
                (12, 26), FONT, 0.5, (150, 205, 255), 1, cv2.LINE_AA)
    return np.vstack([fig, note])


def main():
    ap = argparse.ArgumentParser(description="Drift-Sense pipeline visualiser")
    ap.add_argument("--style", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-px", type=int, default=800,
                    help="die crop shown in the stage figure (layout px)")
    ap.add_argument("--out", default=None,
                    help="output prefix (default: figs/<style>)")
    a = ap.parse_args()

    prefix = a.out or os.path.join("figs", a.style)
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)

    sfig = stage_figure(a.style, a.seed, a.crop_px)
    cv2.imwrite(f"{prefix}_stages.png", sfig)
    print(f"wrote {prefix}_stages.png   {sfig.shape[1]}x{sfig.shape[0]}")

    pfig = pair_figure(a.style, a.seed)
    cv2.imwrite(f"{prefix}_pair.png", pfig)
    print(f"wrote {prefix}_pair.png     {pfig.shape[1]}x{pfig.shape[0]}")


if __name__ == "__main__":
    main()
