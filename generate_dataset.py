#!/usr/bin/env python3
"""
Drift-Sense Synthetic Dataset Generator (DRAM-style)
=====================================================
Generates (Reference, Search) image pairs for the Navigation-Error
Recovery challenge (SEMICON India Hackathon 2026 - Applied Materials track).

Model summary
-------------
1. LAYOUT: large binary DRAM-style layout at high resolution
   - horizontal word-lines + vertical bit-lines + contact via at each
     intersection, with per-cell jitter and rare via defects so the array
     is periodic but not identical cell-to-cell.
2. TWO INDEPENDENT CAPTURES:
   - REFERENCE = small crop imaged at ~100x (sharp, low noise)
   - SEARCH    = whole layout imaged at ~10x, downsampled to 1000x1000
     (reference occupies ~100x100 px inside it)
   Each capture gets its OWN SEM chain and its OWN independent noise.
3. SEM CHAIN per capture: edge brightening -> Gaussian beam blur ->
   Poisson shot noise -> Gaussian read noise -> brightness/contrast drift.
   Search image is noisier + blurrier than reference (lower magnification).
4. DEGRADATION on search: small random rotation + scale jitter.
5. GROUND TRUTH: true center (x, y) of the reference inside the search
   image, written to ground_truth.json and ground_truth.csv.

Usage
-----
    python generate_dataset.py --num-pairs 30 --out dataset
    python generate_dataset.py --num-pairs 3 --out demo --seed 42
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. LAYOUT SYNTHESIS (DRAM-style)
# ---------------------------------------------------------------------------

def draw_dram_layout(rng, size=8000, pitch_wl=None, pitch_bl=None):
    """
    Draw a binary DRAM-style layout on a `size x size` canvas.

    Returns float32 image in [0, 1]:
        background substrate ~0.25, lines ~0.65, vias ~1.0
    """
    if pitch_wl is None:
        pitch_wl = rng.integers(60, 90)     # word-line pitch (px at hi-res)
    if pitch_bl is None:
        pitch_bl = rng.integers(60, 90)     # bit-line pitch

    line_w_wl = max(8, int(pitch_wl * rng.uniform(0.30, 0.42)))
    line_w_bl = max(8, int(pitch_bl * rng.uniform(0.30, 0.42)))
    via_r     = max(5, int(min(pitch_wl, pitch_bl) * rng.uniform(0.14, 0.20)))

    img = np.full((size, size), 0.25, np.float32)          # substrate

    # Cell-to-cell variation must SURVIVE the 10x downsample to the search
    # image, otherwise every cell looks identical and localization becomes
    # mathematically impossible. All variation below is physically grounded:
    #   - line-width (CD) variation across the die   [litho process variation]
    #   - line-edge position jitter                  [overlay error]
    #   - per-via size variation                     [via CD variation]
    #   - missing / bridged vias                     [via open & short defects]

    # horizontal word-lines (per-line width + intensity variation)
    y = int(rng.integers(0, pitch_wl))
    wl_rows = []
    while y < size:
        jitter = int(rng.integers(-4, 5))                  # overlay jitter
        w = max(6, int(line_w_wl * rng.uniform(0.85, 1.15)))   # CD variation
        inten = 0.65 * rng.uniform(0.94, 1.06)                 # film thickness
        y0 = np.clip(y + jitter, 0, size - 1)
        img[y0:y0 + w, :] = inten
        wl_rows.append(y0 + w // 2)
        y += pitch_wl

    # vertical bit-lines
    x = int(rng.integers(0, pitch_bl))
    bl_cols = []
    while x < size:
        jitter = int(rng.integers(-4, 5))
        w = max(6, int(line_w_bl * rng.uniform(0.85, 1.15)))
        inten = 0.65 * rng.uniform(0.94, 1.06)
        x0 = np.clip(x + jitter, 0, size - 1)
        img[:, x0:x0 + w] = np.maximum(img[:, x0:x0 + w], inten)
        bl_cols.append(x0 + w // 2)
        x += pitch_bl

    # contact vias at intersections. Per-via radius/position variation plus
    # rare defects -> the array stays "highly periodic" to the eye, but each
    # neighborhood has a unique fingerprint that survives 10x downsampling.
    p_missing = 0.02        # via open (missing contact)
    p_bridge  = 0.005       # via short (two merged contacts)
    for yy in wl_rows:
        for xx in bl_cols:
            r = rng.random()
            if r < p_missing:
                continue                                   # missing via
            rad = max(3, int(via_r * rng.uniform(0.70, 1.30)))  # via CD var.
            dy = int(rng.integers(-3, 4))
            dx = int(rng.integers(-3, 4))
            if r < p_missing + p_bridge:                   # bridged via
                cv2.ellipse(img, (xx + dx, yy + dy),
                            (rad * 2, rad), 0, 0, 360, 1.0, -1)
            else:
                cv2.circle(img, (xx + dx, yy + dy), rad, 1.0, -1)

    return img


def draw_finfet_layout(rng, size=8000, pitch_fin=None, pitch_gate=None):
    """
    Draw a binary FinFET-style layout on a `size x size` canvas.

    Dense parallel vertical fins, crossed by PERIODIC horizontal poly gate
    rows (as in real multi-fin standard-cell rows), so any local crop -
    including the small reference window - sees 1-2 gate crossings, giving
    genuine 2D localizability (fins alone are translation-invariant along
    their length). Bright gate-fin crossings model the higher effective SE
    yield at the 3D corner where gate wraps the fin.
    Returns float32 image in [0, 1]:
        background substrate ~0.20, fins ~0.50, gates ~0.62, crossings ~0.85
    """
    if pitch_fin is None:
        pitch_fin = rng.integers(60, 85)      # fin pitch (px at hi-res) --
        # must stay well above the Nyquist limit after the ~8x downsample to
        # the search image (matches DRAM word/bit-line pitch scale, ~60-90px)
    if pitch_gate is None:
        pitch_gate = rng.integers(500, 700)   # gate-row pitch: sized so a
        # single ~800px reference crop typically contains 1-2 gate rows

    fin_w = max(4, int(pitch_fin * rng.uniform(0.30, 0.40)))
    gate_w = max(30, int(pitch_gate * rng.uniform(0.10, 0.16)))

    img = np.full((size, size), 0.20, np.float32)          # substrate

    # vertical fins with per-fin CD (width) and position jitter
    x = int(rng.integers(0, pitch_fin))
    fin_cols = []
    while x < size:
        jitter = int(rng.integers(-2, 3))                  # overlay jitter
        w = max(3, int(fin_w * rng.uniform(0.85, 1.15)))    # fin CD variation
        inten = 0.50 * rng.uniform(0.94, 1.06)
        x0 = np.clip(x + jitter, 0, size - 1)
        img[:, x0:x0 + w] = inten
        fin_cols.append((x0, x0 + w))
        x += pitch_fin

    # periodic horizontal poly gate rows (per-row width + position jitter)
    y = int(rng.integers(0, pitch_gate))
    gate_rows = []
    while y < size:
        jitter = int(rng.integers(-3, 4))
        w = max(20, int(gate_w * rng.uniform(0.85, 1.15)))
        y0 = np.clip(y + jitter, 0, size - 1)
        img[y0:y0 + w, :] = np.maximum(img[y0:y0 + w, :], 0.62)
        gate_rows.append((y0, y0 + w))
        y += pitch_gate

    # bright crossings where gate wraps each fin + rare fin-break defects
    p_break = 0.015
    for (x0, x1) in fin_cols:
        if rng.random() < p_break:
            continue                                       # missing/broken fin
        for (y0, y1) in gate_rows:
            r = max(2, int(min(x1 - x0, y1 - y0) * rng.uniform(0.9, 1.15)))
            cx = (x0 + x1) // 2 + int(rng.integers(-1, 2))
            cy = (y0 + y1) // 2 + int(rng.integers(-1, 2))
            cv2.circle(img, (cx, cy), r, 0.85, -1)

    return img


def process_variation_field(rng, size, grid=20, amplitude=0.35):
    """
    Smooth, spatially-correlated intensity modulation field, modeling
    across-die CD (critical-dimension) and etch-uniformity variation from
    litho/etch process non-uniformity [see references.md]. This gives every
    neighborhood of the array a locally unique brightness "fingerprint" that
    survives the 10x downsample -- independent of the specific periodic
    structure (DRAM or FinFET) -- which is what makes local navigation sites
    distinguishable inside an otherwise-repeating pattern.

    Returns a (size x size) float32 field centered at 1.0 with correlation
    length ~ size/grid pixels.
    """
    coarse = rng.normal(1.0, amplitude, size=(grid, grid)).astype(np.float32)
    field = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    return field


LAYOUTS = {"dram": draw_dram_layout, "finfet": draw_finfet_layout}


# ---------------------------------------------------------------------------
# 2. SEM IMAGING CHAIN
# ---------------------------------------------------------------------------

def sem_capture(layout, rng, blur_sigma, dose, read_sigma, edge_gain):
    """
    Simulate one independent SEM capture of `layout` (float32 [0,1]).

    Steps (physical order):
      1. edge brightening  : SE yield increases at topography edges
      2. beam blur         : Gaussian probe PSF
      3. shot noise        : Poisson, dose = mean electrons at signal 1.0
      4. read noise        : additive Gaussian
      5. gain/offset drift : mild brightness & contrast variation
    A fresh noise realization is drawn every call - never reused.
    """
    img = layout.copy()

    # 1. edge brightening
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.5)
    if edges.max() > 0:
        edges /= edges.max()
    img = np.clip(img + edge_gain * edges, 0, 1.4)

    # 2. beam blur
    img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    # 3. Poisson shot noise (independent every call)
    img = rng.poisson(img * dose).astype(np.float32) / dose

    # 4. Gaussian read noise (independent every call)
    img += rng.normal(0.0, read_sigma, img.shape).astype(np.float32)

    # 5. brightness / contrast drift
    gain = rng.uniform(0.92, 1.08)
    offset = rng.uniform(-0.04, 0.04)
    img = img * gain + offset

    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# 3. PAIR GENERATION
# ---------------------------------------------------------------------------

def generate_pair(rng, style="dram", search_size=1000, ref_size=256,
                  scale_ratio=10.0, noise_scale=1.0, pv_amplitude=None):
    """
    Build one (reference, search, ground_truth) sample.

    Returns
    -------
    ref_u8    : reference image  (ref_size x ref_size, uint8)
    search_u8 : search image     (search_size x search_size, uint8)
    gt        : dict with true center (x, y) in search-image pixels + params
    """
    # hi-res layout: search_size at 10x  ->  layout is 10x larger per axis...
    # rendering 10000px is slow, so render at 8000 and treat the mapping
    # search_px = layout_px / layout_to_search
    layout_size = 8000
    layout_fn = LAYOUTS[style]
    layout = layout_fn(rng, size=layout_size)

    # Apply the smooth process-variation fingerprint (see
    # process_variation_field docstring) BEFORE cropping/rendering, so the
    # reference and search captures see the identical underlying modulation.
    #
    # DOMAIN RANDOMIZATION: the amplitude is randomized per pair (including
    # near-zero) rather than fixed. Real dies vary widely in how much
    # across-die CD non-uniformity they exhibit, and the evaluation set is
    # generated independently of ours -- so an algorithm must not assume a
    # strong smooth fingerprint is always present. Training/tuning across the
    # full range prevents over-reliance on this single cue.
    if pv_amplitude is None:
        pv_amplitude = float(rng.uniform(0.0, 0.35))
    pv_field = process_variation_field(rng, layout_size, amplitude=pv_amplitude)
    layout = np.clip(layout * pv_field, 0, 1.4).astype(np.float32)

    layout_to_search = layout_size / search_size            # 8.0 px per search px

    # ---- choose the true site (in layout coordinates) ----
    # reference covers ref_footprint layout px so that in the search image it
    # spans ~ (search_size/scale_ratio) = 100 px
    ref_span_search = search_size / scale_ratio             # ~100 px in search
    ref_footprint = int(ref_span_search * layout_to_search) # layout px (~800)

    margin = ref_footprint // 2 + 50
    cx_l = int(rng.integers(margin, layout_size - margin))  # layout coords
    cy_l = int(rng.integers(margin, layout_size - margin))

    # ---- REFERENCE capture (100x): crop then image ----
    half = ref_footprint // 2
    ref_crop = layout[cy_l - half:cy_l + half, cx_l - half:cx_l + half]
    ref_hi = cv2.resize(ref_crop, (ref_size, ref_size),
                        interpolation=cv2.INTER_AREA)
    ref_img = sem_capture(
        ref_hi, rng,
        blur_sigma=rng.uniform(0.8, 1.2),      # sharp: high mag, small probe
        dose=rng.uniform(180, 260) / noise_scale,   # high dose -> low shot noise
        read_sigma=rng.uniform(0.01, 0.02),
        edge_gain=rng.uniform(0.25, 0.40),
    )

    # ---- SEARCH capture (10x): degrade whole layout, then downsample ----
    # stage/optics imprecision: small rotation + scale jitter of the scene
    angle = rng.uniform(-1.5, 1.5)                       # degrees
    scale_jit = rng.uniform(0.97, 1.03)
    M = cv2.getRotationMatrix2D((layout_size / 2, layout_size / 2),
                                angle, scale_jit)
    layout_deg = cv2.warpAffine(layout, M, (layout_size, layout_size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

    search_hi = cv2.resize(layout_deg, (search_size, search_size),
                           interpolation=cv2.INTER_AREA)
    search_img = sem_capture(
        search_hi, rng,
        blur_sigma=rng.uniform(1.2, 1.8),      # blurrier: low mag
        dose=rng.uniform(50, 90) / noise_scale,     # low dose -> strong shot noise
        read_sigma=rng.uniform(0.03, 0.05),
        edge_gain=rng.uniform(0.20, 0.35),
    )

    # ---- ground truth: map (cx_l, cy_l) through the affine, then downscale ----
    pt = M @ np.array([cx_l, cy_l, 1.0])
    cx_s = float(pt[0] / layout_to_search)
    cy_s = float(pt[1] / layout_to_search)

    gt = {
        "style": style,
        "pv_amplitude": round(pv_amplitude, 4),
        "true_center_x": round(cx_s, 2),
        "true_center_y": round(cy_s, 2),
        "ref_span_in_search_px": round(ref_span_search, 1),
        "rotation_deg": round(angle, 3),
        "scale_jitter": round(scale_jit, 4),
    }
    return (np.uint8(ref_img * 255), np.uint8(search_img * 255), gt)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Drift-Sense synthetic dataset generator")
    ap.add_argument("--num-pairs", type=int, default=30)
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--style", type=str, default="dram",
                    choices=["dram", "finfet", "mixed"],
                    help="architecture style: dram, finfet, or mixed (alternate)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--search-size", type=int, default=1000)
    ap.add_argument("--ref-size", type=int, default=256)
    ap.add_argument("--pv-amplitude", type=float, default=None,
                    help="fix the process-variation fingerprint amplitude "
                         "(default: randomized per pair in [0, 0.35])")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help=">1 = noisier images (stress testing)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    records = []
    for i in range(args.num_pairs):
        style = args.style if args.style != "mixed" else ("dram" if i % 2 == 0 else "finfet")
        ref, search, gt = generate_pair(rng, style=style,
                                        search_size=args.search_size,
                                        ref_size=args.ref_size,
                                        noise_scale=args.noise_scale,
                                        pv_amplitude=args.pv_amplitude)
        rname = f"pair{i:03d}_reference.png"
        sname = f"pair{i:03d}_search.png"
        cv2.imwrite(os.path.join(args.out, rname), ref)
        cv2.imwrite(os.path.join(args.out, sname), search)
        gt.update({"pair_id": i, "reference": rname, "search": sname})
        records.append(gt)
        print(f"[{i + 1}/{args.num_pairs}] ({style}) true center = "
              f"({gt['true_center_x']}, {gt['true_center_y']})")

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(records, f, indent=2)
    keys = ["pair_id", "style", "pv_amplitude", "reference", "search", "true_center_x",
            "true_center_y", "ref_span_in_search_px",
            "rotation_deg", "scale_jitter"]
    with open(os.path.join(args.out, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)

    print(f"\nDone. {args.num_pairs} pairs written to '{args.out}/' "
          f"with ground_truth.json / .csv")


if __name__ == "__main__":
    main()
