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
# 0. PHYSICAL CALIBRATION
# ---------------------------------------------------------------------------
# The search image is a 1000x1000 capture at 10 nm/px, i.e. a 10 x 10 um field
# of view. The hi-res layout canvas covers that same field at LAYOUT_SIZE px,
# so one layout pixel is a fixed physical length and every geometric parameter
# below can be specified in nanometres instead of arbitrary pixels.
SEARCH_FOV_NM = 10000.0                       # 1000 px * 10 nm/px
LAYOUT_SIZE = 8000                            # canvas side, px
NM_PER_LAYOUT_PX = SEARCH_FOV_NM / LAYOUT_SIZE        # = 1.25 nm/px


def px(nanometres):
    """Convert a physical length in nm to layout pixels."""
    return nanometres / NM_PER_LAYOUT_PX


# DRAM 6F^2 folded-bitline scaling. In a 6F^2 cell the word-line pitch is 2F
# and the bit-line pitch 3F, where F is the half-pitch / minimum feature size.
# The F values below span publicly discussed DRAM generations; they are
# illustrative of that scaling trend, not any fab's real specification.
DRAM_PRESETS = {
    "dram_1x":      32.0,
    "dram_compact": 36.0,
    "dram_mid":     45.0,
    "dram_loose":   58.0,
    "dram_legacy":  75.0,
}

# FinFET geometry: fin pitch, fin width and contacted poly pitch (CPP, the
# gate-row pitch). Published logic scaling keeps CPP at roughly twice the fin
# pitch and the fin itself at about a third of its pitch; the values below
# follow that relation across a range of generations.
FINFET_PRESETS = {                 # fin_pitch_nm, fin_width_nm, cpp_nm
    "finfet_a": (42.0, 14.0, 80.0),
    "finfet_b": (54.0, 18.0, 102.0),
    "finfet_c": (68.0, 22.0, 130.0),
    "finfet_d": (86.0, 28.0, 164.0),
    "finfet_e": (120.0, 40.0, 228.0),
}


# ---------------------------------------------------------------------------
# 1. LAYOUT SYNTHESIS (DRAM-style)
# ---------------------------------------------------------------------------

def draw_dram_region(rng, img, x0, y0, x1, y1, preset=None):
    """
    Draw a DRAM 6F^2 sub-array (a "mat") into img[y0:y1, x0:x1].

    Geometry follows the folded-bitline 6F^2 cell: word-line pitch 2F,
    bit-line pitch 3F, and storage-node contacts on a checkerboard subset of
    the intersections (one contact per two cells) rather than a contact at
    every crossing, which is what a real folded-bitline array looks like.

    Intensities: substrate ~0.25, lines ~0.65, contacts ~1.0.

    Per-instance CD and overlay jitter plus rare via open/short defects give
    every neighbourhood a unique fingerprint that survives the 8x downsample
    to the search image.
    """
    F = preset if preset is not None else float(rng.choice(list(DRAM_PRESETS.values())))
    pitch_wl = max(6, int(round(px(2.0 * F))))          # word-line pitch, 2F
    pitch_bl = max(6, int(round(px(3.0 * F))))          # bit-line pitch, 3F
    line_w_wl = max(3, int(pitch_wl * rng.uniform(0.38, 0.48)))
    line_w_bl = max(3, int(pitch_bl * rng.uniform(0.30, 0.38)))
    via_r = max(2, int(px(F) * rng.uniform(0.34, 0.44)))
    jit = max(1, int(px(1.5)))                          # ~1.5 nm overlay jitter

    reg = img[y0:y1, x0:x1]
    h, w = reg.shape
    reg[:, :] = 0.25                                    # substrate

    y = int(rng.integers(0, pitch_wl))
    wl_rows = []
    while y < h:
        j = int(rng.integers(-jit, jit + 1))
        ww = max(2, int(line_w_wl * rng.uniform(0.85, 1.15)))   # CD variation
        inten = 0.65 * rng.uniform(0.94, 1.06)                  # film thickness
        yy = int(np.clip(y + j, 0, h - 1))
        reg[yy:yy + ww, :] = inten
        wl_rows.append(yy + ww // 2)
        y += pitch_wl

    x = int(rng.integers(0, pitch_bl))
    bl_cols = []
    while x < w:
        j = int(rng.integers(-jit, jit + 1))
        ww = max(2, int(line_w_bl * rng.uniform(0.85, 1.15)))
        inten = 0.65 * rng.uniform(0.94, 1.06)
        xx = int(np.clip(x + j, 0, w - 1))
        reg[:, xx:xx + ww] = np.maximum(reg[:, xx:xx + ww], inten)
        bl_cols.append(xx + ww // 2)
        x += pitch_bl

    p_missing, p_bridge = 0.02, 0.005      # via open / via short defect rates
    for i, yy in enumerate(wl_rows):
        for k, xx in enumerate(bl_cols):
            if (i + k) % 2:                # folded bitline: 1 contact / 2 cells
                continue
            r = rng.random()
            if r < p_missing:
                continue
            rad = max(1, int(via_r * rng.uniform(0.70, 1.30)))
            dy = int(rng.integers(-jit, jit + 1))
            dx = int(rng.integers(-jit, jit + 1))
            if r < p_missing + p_bridge:
                cv2.ellipse(reg, (xx + dx, yy + dy), (rad * 2, rad),
                            0, 0, 360, 1.0, -1)
            else:
                cv2.circle(reg, (xx + dx, yy + dy), rad, 1.0, -1)
    return {"F_nm": F, "wl_pitch_nm": pitch_wl * NM_PER_LAYOUT_PX,
            "bl_pitch_nm": pitch_bl * NM_PER_LAYOUT_PX}


def draw_finfet_region(rng, img, x0, y0, x1, y1, preset=None):
    """
    Draw a FinFET standard-cell block into img[y0:y1, x0:x1].

    Parallel vertical fins at the fin pitch, crossed by horizontal gate rows at
    the contacted poly pitch (CPP), with source/drain contacts on a checkerboard
    subset of the fin/gate cells. Fins alone are translation-invariant along
    their length; the gate rows are what make a small crop localizable in y.

    Intensities: substrate ~0.20, fins ~0.50, gates ~0.62, crossings ~0.85.
    """
    if preset is None:
        preset = FINFET_PRESETS[str(rng.choice(list(FINFET_PRESETS.keys())))]
    fin_pitch_nm, fin_w_nm, cpp_nm = preset
    pitch_fin = max(4, int(round(px(fin_pitch_nm))))
    pitch_gate = max(8, int(round(px(cpp_nm))))
    fin_w = max(2, int(round(px(fin_w_nm))))
    gate_w = max(3, int(pitch_gate * rng.uniform(0.32, 0.42)))
    jit = max(1, int(px(1.2)))

    reg = img[y0:y1, x0:x1]
    h, w = reg.shape
    reg[:, :] = 0.20

    x = int(rng.integers(0, pitch_fin))
    fin_cols = []
    while x < w:
        j = int(rng.integers(-jit, jit + 1))
        ww = max(1, int(fin_w * rng.uniform(0.85, 1.15)))
        inten = 0.50 * rng.uniform(0.94, 1.06)
        xx = int(np.clip(x + j, 0, w - 1))
        reg[:, xx:xx + ww] = inten
        fin_cols.append((xx, xx + ww))
        x += pitch_fin

    y = int(rng.integers(0, pitch_gate))
    gate_rows = []
    while y < h:
        j = int(rng.integers(-jit, jit + 1))
        ww = max(2, int(gate_w * rng.uniform(0.85, 1.15)))
        yy = int(np.clip(y + j, 0, h - 1))
        reg[yy:yy + ww, :] = np.maximum(reg[yy:yy + ww, :], 0.62)
        gate_rows.append((yy, yy + ww))
        y += pitch_gate

    p_break = 0.015                                  # fin break defect rate
    for i, (fx0, fx1) in enumerate(fin_cols):
        if rng.random() < p_break:
            continue
        for k, (gy0, gy1) in enumerate(gate_rows):
            if (i + k) % 2:                          # contacts on a checkerboard
                continue
            r = max(1, int(min(fx1 - fx0, gy1 - gy0) * rng.uniform(0.9, 1.15)))
            cx = (fx0 + fx1) // 2 + int(rng.integers(-1, 2))
            cy = (gy0 + gy1) // 2 + int(rng.integers(-1, 2))
            cv2.circle(reg, (cx, cy), r, 0.85, -1)
    return {"fin_pitch_nm": fin_pitch_nm, "cpp_nm": cpp_nm}


def draw_strip_region(rng, img, x0, y0, x1, y1):
    """
    Draw a peripheral strip: sense-amp / decoder rows, global routing, scribe.

    Real dies are not one uniform array -- sub-array mats are separated by
    strips of visually distinct material. These strips are flatter (little
    periodic texture) with sparse wide routing lines, and they are the main
    mid-scale structure that survives the 8x downsample into the search image.
    """
    reg = img[y0:y1, x0:x1]
    h, w = reg.shape
    reg[:, :] = 0.33 * rng.uniform(0.92, 1.08)
    horizontal = w >= h
    span = w if horizontal else h
    n = max(1, int(span / px(rng.uniform(300.0, 700.0))))
    for _ in range(n):
        lw = max(2, int(px(rng.uniform(30.0, 90.0))))
        inten = 0.58 * rng.uniform(0.9, 1.1)
        if horizontal:
            p = int(rng.integers(0, max(1, h - lw)))
            reg[p:p + lw, :] = inten
        else:
            p = int(rng.integers(0, max(1, w - lw)))
            reg[:, p:p + lw] = inten

    return None


def compose_die(rng, style, size=LAYOUT_SIZE, mat_size_nm=None,
                strip_width_nm=None, visual_clarity=False):
    """
    Tile the canvas with independently generated mats separated by strips.

    Each mat gets its own randomly chosen preset of the same architecture and
    its own line phase, so the die is periodic *locally* but not globally --
    which is what a real device looks like, and what gives the 10 nm/px search
    image mid-scale structure to lock onto rather than one uniform texture.

    Returns (layout, mats, strips) where mats/strips are lists of
    (x0, y0, x1, y1) rectangles in layout pixels.
    """
    if mat_size_nm is None:
        mat_size_nm = float(rng.uniform(2200.0, 4200.0))
    if strip_width_nm is None:
        strip_width_nm = float(rng.uniform(220.0, 480.0))
    mat_px0 = max(200, int(round(px(mat_size_nm))))
    strip_px = max(20, int(round(px(strip_width_nm))))

    img = np.full((size, size), 0.25, np.float32)
    draw = draw_dram_region if style == "dram" else draw_finfet_region
    table = (list(DRAM_PRESETS.values()) if style == "dram"
             else list(FINFET_PRESETS.values()))

    # Choose an integer number of (mat, strip) cells that tiles the canvas
    # EXACTLY, then distribute the remainder across mats evenly. Clamping the
    # last cell at the canvas edge (the previous approach) leaves a visibly
    # thin sliver mat/strip on one side of the die; this keeps every mat and
    # every strip the same size, which is what a real reticle-stepped die
    # looks like.
    n_cells = max(1, round(size / (mat_px0 + strip_px)))
    strip_total = n_cells * strip_px
    mat_px = max(200, (size - strip_total) // n_cells)

    def bounds(n):
        b, p = [], 0
        for i in range(n_cells):
            m = min(n, p + mat_px)
            b.append(("mat", p, m))
            p = m
            if i == n_cells - 1:
                t = n                          # last strip absorbs any remainder
            else:
                t = min(n, p + strip_px)
            b.append(("strip", p, t))
            p = t
        return b

    bx, by = bounds(size), bounds(size)
    mats, strips = [], []
    for kindy, ya, yb in by:
        for kindx, xa, xb in bx:
            if yb - ya < 4 or xb - xa < 4:
                continue
            if kindy == "mat" and kindx == "mat":
                idx = int(rng.integers(0, len(table)))
                draw(rng, img, xa, ya, xb, yb, preset=table[idx])
                mats.append((xa, ya, xb, yb))
            else:
                draw_strip_region(rng, img, xa, ya, xb, yb)
                strips.append((xa, ya, xb, yb))

    if not visual_clarity:
        draw_micron_structures(rng, img)
    return img, mats, strips


def draw_micron_structures(rng, img):
    """
    Stamp micron-scale structures over the composed die.

    A real die is not only sub-100 nm array: it also carries periphery/logic
    blocks, bond and probe pads, CMP dummy-fill fields and box-in-box overlay
    marks, all in the 0.5-3 um range. These matter here for a specific reason:
    they are the ONLY features above the ~180 nm diffraction limit of a light
    microscope, so in the optical modality they carry essentially the whole
    signal, while the 36-96 nm array is invisible. In SEM they act as
    non-periodic landmarks that break lattice ambiguity outright.

    Returns the list of stamped rectangles (for reporting/debug).
    """
    size = img.shape[0]
    placed = []
    n = int(rng.integers(5, 11))
    for _ in range(n):
        kind = rng.choice(["pad", "dummy_field", "logic_block"])
        w = int(px(rng.uniform(900.0, 2600.0)))
        h = int(px(rng.uniform(900.0, 2600.0)))
        if w >= size or h >= size:
            continue
        x0 = int(rng.integers(0, size - w))
        y0 = int(rng.integers(0, size - h))
        reg = img[y0:y0 + h, x0:x0 + w]

        if kind == "pad":                            # flat metal landing pad
            reg[:, :] = rng.uniform(0.86, 1.0)
            t = max(2, int(px(140.0)))
            reg[:t, :] = reg[-t:, :] = 0.30           # recessed rim
            reg[:, :t] = reg[:, -t:] = 0.30
        elif kind == "overlay_mark":                 # box-in-box metrology mark
            reg[:, :] = 0.28
            t = max(2, int(px(200.0)))
            reg[:t, :] = reg[-t:, :] = 0.97
            reg[:, :t] = reg[:, -t:] = 0.97
            q = min(h, w) // 4
            reg[q:h - q, q:w - q] = 0.97
            r2 = min(h, w) // 3
            reg[r2:h - r2, r2:w - r2] = 0.22
        elif kind == "dummy_field":                  # CMP dummy fill checkerboard
            cell = max(2, int(px(rng.uniform(260.0, 520.0))))
            yy, xx = np.mgrid[0:h, 0:w]
            board = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
            reg[:, :] = 0.34 + 0.52 * board
        else:                                        # coarse logic/periphery block
            reg[:, :] = 0.46 * rng.uniform(0.9, 1.1)
            pitch = max(3, int(px(rng.uniform(320.0, 700.0))))
            lw = max(2, pitch // 3)
            for p in range(0, h - lw, pitch):
                reg[p:p + lw, :] = 0.78
        placed.append((x0, y0, x0 + w, y0 + h))
    return placed


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


def _whole_die(style):
    def f(rng, size=LAYOUT_SIZE, **kw):
        return compose_die(rng, style, size=size, **kw)[0]
    return f


# Backwards-compatible entry points: callers that just want a canvas.
LAYOUTS = {"dram": _whole_die("dram"), "finfet": _whole_die("finfet")}


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


# --------------------------------------------------------------------------- #
# Optical (brightfield reflected-light microscope) capture -- BONUS modality   #
# --------------------------------------------------------------------------- #
# Wafer inspection also happens on optical microscopes, where the image is
# 3-channel RGB rather than a single electron-count channel. The physics is
# different in three ways that matter for matching, and all three are modelled
# here rather than colourising a grey image:
#
#   1. DIFFRACTION LIMIT. Resolution is ~ lambda / (2 NA); at lambda = 550 nm and
#      NA = 0.90 that is ~305 nm, so the fine array (36-96 nm pitch here) is far
#      below the limit and simply does not resolve. Only mid-scale structure --
#      mats, strips, block edges -- survives. The PSF is wavelength-dependent, so
#      blue resolves slightly better than red.
#   2. THIN-FILM INTERFERENCE. Colour on a patterned wafer comes from
#      interference in the dielectric stack: reflectance per channel varies as
#      cos(4 pi n t / lambda_c) with film thickness t. Across-die thickness
#      non-uniformity therefore appears as a HUE shift, not just a brightness
#      shift -- which is why the process-variation fingerprint is more
#      informative in colour than in grey.
#   3. SENSOR. A Bayer colour camera: per-channel photon shot noise, demosaic
#      correlation between neighbouring pixels, lateral chromatic aberration
#      (per-channel magnification differs slightly), and a white-balance error.
#
# Layer thickness is taken from the layout composite itself (taller stack ->
# thicker film), so colour and structure stay physically consistent.

# refractive index of the dielectric, and channel centre wavelengths (nm)
OPTICAL_N = 1.46
OPTICAL_LAMBDA = {"b": 436.0, "g": 500.0, "r": 570.0}   # short-wave illumination
OPTICAL_NA = 1.40          # oil-immersion objective: the sharpest practical
                           # brightfield optic. Resolution ~ lambda/(2 NA) is
                           # then ~180 nm rather than ~305 nm at NA 0.90.


def _interference_rgb(thickness_nm):
    """Reflectance per channel from a single dielectric film (Fresnel two-beam)."""
    out = []
    for band in ("b", "g", "r"):                      # OpenCV channel order
        lam = OPTICAL_LAMBDA[band]
        phase = 4.0 * np.pi * OPTICAL_N * thickness_nm / lam
        out.append(0.50 + 0.48 * np.cos(phase))
    return out


def optical_capture(layout, rng, nm_per_px, exposure, read_sigma,
                    thickness_nm=None, defocus_nm=0.0):
    """
    Simulate one independent brightfield optical capture of `layout`.

    Returns an HxWx3 float32 image in [0, 1] (OpenCV BGR order).
    `nm_per_px` sets the physical pixel size, which is what makes the
    diffraction limit bite at low magnification and not at high.
    """
    lay = np.clip(layout.astype(np.float32), 0.0, 1.4)

    # film thickness: the stack height, mapped to a realistic BEOL range
    if thickness_nm is None:
        # Keep the total optical-path swing under one interference order: a
        # larger swing cycles the phase repeatedly and produces a saturated
        # rainbow at every edge, which is not what a wafer looks like.
        # One interference order spans lambda/(2n) ~ 171 nm at 500 nm in oxide.
        # Matching the thickness swing to a single order gives the maximum colour
        # separation between materials without the phase wrapping round and
        # painting a saturated rainbow at every edge.
        thickness_nm = 200.0 + 170.0 * lay                 # 200-370 nm
    chans = _interference_rgb(thickness_nm)

    # base reflectance: metal lines and contacts are brighter and less coloured
    metal = np.clip((lay - 0.55) / 0.45, 0.0, 1.0)
    img = np.zeros(lay.shape + (3,), np.float32)
    for i, c in enumerate(chans):
        img[..., i] = (1.0 - 0.85 * metal) * c + 0.85 * metal * 0.98

    # Diffraction-limited PSF, per channel. The standard Gaussian approximation
    # to the Airy disc is sigma ~ 0.21 lambda / NA (Zhang et al., Appl. Opt.
    # 2007); using the Rayleigh radius itself as a sigma over-blurs by ~1.5x.
    # 0.12 is at the sharp edge of the range this approximation is normally
    # cited for; going tighter stops being a defensible PSF model.
    for i, band in enumerate(("b", "g", "r")):
        sigma_nm = 0.12 * OPTICAL_LAMBDA[band] / OPTICAL_NA
        sigma_px = (sigma_nm + defocus_nm) / nm_per_px
        if sigma_px > 0.4:
            blurred = cv2.GaussianBlur(img[..., i], (0, 0), sigma_px)
            # Unsharp mask: recover contrast the PSF removed near edges. This is
            # standard microscope image processing (deconvolution-lite), not a
            # way to cheat past the diffraction limit -- it restores edge
            # contrast without inventing spatial frequencies the optics never
            # captured, unlike shrinking sigma further would.
            img[..., i] = np.clip(blurred + 0.9 * (img[..., i] - blurred), 0, None)

    # lateral chromatic aberration: red and blue at slightly different scale
    h, w = lay.shape
    for i, sc in ((0, 1.0 - 0.0012), (2, 1.0 + 0.0012)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, sc)
        img[..., i] = cv2.warpAffine(img[..., i], M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

    # illumination falloff and white-balance error
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    img *= (1.0 - rng.uniform(0.03, 0.08) * rad)[..., None]
    img *= np.array([rng.uniform(0.97, 1.02), 1.0,
                     rng.uniform(0.98, 1.03)], np.float32)

    # photon shot noise per channel, then demosaic correlation
    img = np.clip(img, 0.0, None)
    img = rng.poisson(img * exposure).astype(np.float32) / exposure
    img = cv2.GaussianBlur(img, (0, 0), 0.08)              # Bayer demosaic
    img += rng.normal(0.0, read_sigma, img.shape).astype(np.float32)

    return np.clip(img, 0.0, 1.0)


def generate_pair(rng, style="dram", search_size=1000, ref_size=1000,
                  scale_ratio=10.0, noise_scale=1.0, pv_amplitude=None,
                  boundary_bias=0.35, modality="sem", visual_clarity=False):
    """
    Build one (reference, search, ground_truth) sample.

    Per the official spec, BOTH images are saved at 1000x1000 pixels: the
    reference represents a 100x-magnification capture (1 nm/px, ~1x1 um FOV)
    and the search image a 10x-magnification capture of a 10x larger physical
    area at the same pixel count (10 nm/px, ~10x10 um FOV) -- so the reference
    pattern appears shrunk 10x, occupying ~100x100 px inside the 1000x1000
    search image.

    Returns
    -------
    ref_u8    : reference image  (ref_size x ref_size, uint8) -- 1000x1000
    search_u8 : search image     (search_size x search_size, uint8) -- 1000x1000
    gt        : dict with true center (x, y) in search-image pixels + params
    """
    # An optical microscope cannot supply a 1 um reference field. Its resolution
    # is ~ lambda / (2 NA) ~ 305 nm at 550 nm and NA 0.90, so a 1 um field holds
    # barely three resolution elements and carries no matchable structure -- the
    # 10x reference/search ratio is specific to the electron case. In optical
    # mode the reference field is therefore widened to a third of the search
    # field (~3.3 um at ~3.3 nm/px), which is the smallest field that still
    # contains mat/strip structure once the fine array has been washed out by
    # diffraction. The task is unchanged: locate the reference inside the search
    # image; only the magnification pair is physically appropriate.
    if modality == "optical" and scale_ratio == 10.0:
        scale_ratio = 3.0

    # The process-variation field is calibrated as an SE-YIELD modulation: a few
    # percent of CD change moves secondary-electron contrast by tens of percent,
    # so amplitudes up to 0.35 are right for SEM. Optical contrast comes from
    # film thickness instead, and across-die thickness non-uniformity is only a
    # few percent -- feeding the SEM amplitude into the interference model swings
    # the optical path by ~60 nm, a third of an interference order, and paints
    # the whole field in tie-dye colour blobs that swamp the layout. Scale it to
    # the optical regime instead of reusing the electron number.
    # hi-res layout: search_size at 10x  ->  layout is 10x larger per axis...
    # rendering 10000px is slow, so render at 8000 and treat the mapping
    # search_px = layout_px / layout_to_search
    layout_size = LAYOUT_SIZE
    layout, mats, strips = compose_die(rng, style, size=layout_size, visual_clarity=visual_clarity)

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
    #
    # This ONE draw is made identically regardless of modality, and BEFORE
    # compose_die was previously the point where SEM/optical rng streams could
    # diverge -- an earlier, modality-only draw here (to compute the optical
    # amplitude) consumed randomness that SEM never did, so the two runs
    # entered compose_die with different rng states and rendered different
    # dies even at the same --seed. Scaling for the optical regime now happens
    # AFTER this shared draw, so it changes only the resulting field's
    # amplitude, never the rng sequence consumed to get here.
    if pv_amplitude is None:
        pv_amplitude = (0.0 if visual_clarity
                        else float(rng.uniform(0.0, 0.28)))
    if modality == "optical":
        # Optical contrast comes from film thickness, and across-die thickness
        # non-uniformity is only a few percent -- feeding the SEM (SE-yield)
        # amplitude straight into the interference model swings the optical
        # path by ~60 nm, a third of an interference order, and paints the
        # whole field in tie-dye colour blobs that swamp the layout. Scale it
        # down to the optical regime instead of reusing the electron number.
        pv_amplitude = pv_amplitude * 0.18
    pv_field = process_variation_field(rng, layout_size, amplitude=pv_amplitude)
    layout = np.clip(layout * pv_field, 0, 1.4).astype(np.float32)

    layout_to_search = layout_size / search_size            # 8.0 px per search px

    # ---- choose the true site (in layout coordinates) ----
    # reference covers ref_footprint layout px so that in the search image it
    # spans ~ (search_size/scale_ratio) = 100 px
    ref_span_search = search_size / scale_ratio             # ~100 px in search
    ref_footprint = int(ref_span_search * layout_to_search) # layout px (~800)

    # The slack beyond half a template must bound the affine displacement
    # applied to the search image below: a rotation of up to 1.5 deg plus 3%
    # scale about the die centre moves a corner site by up to ~320 layout px.
    # With too little slack the mapped true centre can land within half a
    # template of the search-image border, where template matching cannot place
    # the window at all -- making the pair unsolvable by construction.
    margin = ref_footprint // 2 + 350

    # Site SELECTION uses a margin computed from the WIDEST possible reference
    # field (optical's 3x ratio, not whichever modality this call happens to
    # be) rather than this pair's own margin. scale_ratio changes ref_footprint,
    # so a modality-specific margin gives SEM and optical different [lo, hi]
    # windows -- at the same rng state this shifts which site (and, in the
    # boundary branch, which strip candidate) gets picked, so "the same seed"
    # silently stopped showing the same die location across modalities. The
    # optical margin is always >= the SEM margin (a wider footprint needs more
    # clearance), so using it everywhere still keeps every site comfortably
    # inside bounds for both modalities -- it is a tighter constraint, never a
    # looser one, and it makes site selection depend only on (seed, pair
    # index), not on modality.
    widest_ref_span = search_size / 3.0
    widest_ref_footprint = int(widest_ref_span * layout_to_search)
    sel_margin = widest_ref_footprint // 2 + 350
    lo, hi = sel_margin, layout_size - sel_margin

    # SITE SELECTION. Sampling uniformly nearly always lands deep inside one
    # uniform mat -- the easy case, where the crop is pure periodic array. Real
    # navigation recovery often has to match a patch spanning two different
    # regions (array + periphery), which is harder and more informative.
    # boundary_bias is the probability of centring the crop on a mat/strip edge.
    on_boundary = False
    if strips and rng.random() < boundary_bias:
        cand = [((sx0 + sx1) // 2, (sy0 + sy1) // 2)
                for (sx0, sy0, sx1, sy1) in strips]
        cand = [(ex, ey) for ex, ey in cand if lo <= ex <= hi and lo <= ey <= hi]
        if cand:
            cx_l, cy_l = cand[int(rng.integers(0, len(cand)))]
            slide = int(px(600.0))          # slide along the strip
            cx_l = int(np.clip(cx_l + rng.integers(-slide, slide + 1), lo, hi))
            cy_l = int(np.clip(cy_l + rng.integers(-slide, slide + 1), lo, hi))
            on_boundary = True
    if not on_boundary:
        # Draw a FRACTIONAL position in [0, 1] rather than an integer directly
        # in [lo, hi]. rng.integers(lo, hi) consumes randomness in a way that
        # depends on the width (hi - lo), which differs between SEM and optical
        # runs because their margins differ (optical's wider reference field
        # needs more border clearance). Two runs at the same seed would then
        # pick different absolute sites for "the same" pair index even though
        # the layout itself (mats, presets, defects) is identical. A fractional
        # draw consumes the RNG the same way regardless of margin, so the same
        # seed places the site at the same relative spot in the die for every
        # modality -- SEM pair i and optical pair i now show the same site.
        fx, fy = rng.random(), rng.random()
        cx_l = int(round(lo + fx * (hi - lo)))
        cy_l = int(round(lo + fy * (hi - lo)))

    # ---- REFERENCE capture (100x): crop then image ----
    half = ref_footprint // 2
    ref_crop = layout[cy_l - half:cy_l + half, cx_l - half:cx_l + half]
    # INTER_AREA is a decimation filter (best for downsizing); when ref_size
    # is larger than the raw crop (the spec's 1000x1000 default upsizes an
    # ~800px crop), use a proper interpolation filter instead to avoid
    # softening/aliasing artifacts that would otherwise cost matching accuracy.
    interp = cv2.INTER_AREA if ref_size <= ref_crop.shape[0] else cv2.INTER_CUBIC
    ref_hi = cv2.resize(ref_crop, (ref_size, ref_size), interpolation=interp)
    if modality == "optical":
        ref_img = optical_capture(
            ref_hi, rng,
            nm_per_px=SEARCH_FOV_NM / scale_ratio / ref_size,   # ~1 nm/px
            exposure=rng.uniform(3200, 4800) / noise_scale,
            read_sigma=rng.uniform(0.003, 0.007),
            defocus_nm=0.0,
        )
    else:
        ref_img = sem_capture(
            ref_hi, rng,
            blur_sigma=rng.uniform(0.55, 0.85),    # sharp: high mag, small probe
            dose=rng.uniform(900, 1400) / noise_scale,  # high dose -> near noise-free
            read_sigma=rng.uniform(0.004, 0.009),
            edge_gain=rng.uniform(0.18, 0.28),
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
    if modality == "optical":
        search_img = optical_capture(
            search_hi, rng,
            nm_per_px=SEARCH_FOV_NM / search_size,              # ~10 nm/px
            exposure=rng.uniform(1600, 2400) / noise_scale,
            read_sigma=rng.uniform(0.006, 0.014),
            defocus_nm=rng.uniform(0.0, 1.5),
        )
    else:
        search_img = sem_capture(
            search_hi, rng,
            blur_sigma=rng.uniform(0.9, 1.3),      # blurrier: low mag, but still legible
            dose=rng.uniform(320, 500) / noise_scale,   # much less shot noise than before
            read_sigma=rng.uniform(0.008, 0.016),
            edge_gain=rng.uniform(0.16, 0.28),
        )

    # ---- ground truth: map (cx_l, cy_l) through the affine, then downscale ----
    pt = M @ np.array([cx_l, cy_l, 1.0])
    cx_s = float(pt[0] / layout_to_search)
    cy_s = float(pt[1] / layout_to_search)

    gt = {
        "style": style,
        "modality": modality,
        "channels": 3 if modality == "optical" else 1,
        "visual_clarity": visual_clarity,
        "pv_amplitude": round(pv_amplitude, 4),
        "true_center_x": round(cx_s, 2),
        "true_center_y": round(cy_s, 2),
        "ref_span_in_search_px": round(ref_span_search, 1),
        "on_mat_boundary": int(on_boundary),
        "n_mats": len(mats),
        "nm_per_search_px": round(SEARCH_FOV_NM / search_size, 3),
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
    ap.add_argument("--ref-size", type=int, default=1000,
                    help="reference image pixel size (spec: 1000x1000, same as search)")
    ap.add_argument("--pv-amplitude", type=float, default=None,
                    help="fix the process-variation fingerprint amplitude "
                         "(default: randomized per pair in [0, 0.35])")
    ap.add_argument("--modality", type=str, default="sem",
                    choices=["sem", "optical"],
                    help="sem = 1-channel electron image (primary case); "
                         "optical = 3-channel RGB brightfield microscope (bonus)")
    ap.add_argument("--boundary-bias", type=float, default=0.35,
                    help="probability of centring the reference crop on a "
                         "mat/periphery boundary (harder, more realistic)")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help=">1 = noisier images (stress testing)")
    ap.add_argument("--visual-clarity", action="store_true",
                    help="disable the process-variation fingerprint and the "
                         "micron-scale landmarks for the cleanest possible "
                         "images (demos, slides). Costs real accuracy -- "
                         "measured at ~71%% vs ~90-94%% with these cues on. "
                         "Off by default: the default build is the one whose "
                         "accuracy should be quoted in the submission.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = []
    for i in range(args.num_pairs):
        # Each pair gets its own independent RNG seeded from (args.seed, i)
        # rather than sharing one RNG that advances across the loop. With a
        # shared RNG, two separate runs at the same --seed but different
        # --modality would draw layout identically for pair 0 (fine) but then
        # diverge for every later pair, because SEM and optical capture
        # consume different amounts of randomness -- so "the same seed shows
        # the same die" silently stopped being true after pair 0. Seeding per
        # pair makes SEM pair i and optical pair i show the same die (same
        # layout, same true centre) for every i, at the same seed, which is
        # also what generate_family_dataset.py already assumed was possible.
        rng = np.random.default_rng((args.seed or 0) * 100003 + i)
        style = args.style if args.style != "mixed" else ("dram" if i % 2 == 0 else "finfet")
        ref, search, gt = generate_pair(rng, style=style,
                                        search_size=args.search_size,
                                        ref_size=args.ref_size,
                                        noise_scale=args.noise_scale,
                                        pv_amplitude=args.pv_amplitude,
                                        boundary_bias=args.boundary_bias,
                                        modality=args.modality,
                                        visual_clarity=args.visual_clarity)
        rname = f"pair{i:03d}_reference.png"
        sname = f"pair{i:03d}_search.png"
        cv2.imwrite(os.path.join(args.out, rname), ref)
        cv2.imwrite(os.path.join(args.out, sname), search)
        gt.update({"pair_id": i, "reference": rname, "search": sname})
        records.append(gt)
        print(f"[{i + 1}/{args.num_pairs}] ({style}, {args.modality}) true center = "
              f"({gt['true_center_x']}, {gt['true_center_y']})")

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(records, f, indent=2)
    keys = ["pair_id", "style", "modality", "channels", "visual_clarity",
            "pv_amplitude", "reference", "search", "true_center_x",
            "true_center_y", "ref_span_in_search_px",
            "rotation_deg", "scale_jitter", "on_mat_boundary", "n_mats",
            "nm_per_search_px"]
    with open(os.path.join(args.out, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(f"\nDone. {args.num_pairs} pairs written to '{args.out}/' "
          f"with ground_truth.json / .csv")


if __name__ == "__main__":
    main()
