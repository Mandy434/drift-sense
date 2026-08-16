#!/usr/bin/env python3
"""
Drift-Sense sample-family dataset generator
===========================================
Companion to generate_dataset.py.  Instead of one reference per layout, this
builds a FAMILY: one die (one layout + one process-variation field + one search
capture) and several reference sites cropped from that same die.

Why families matter
-------------------
generate_dataset.py draws a fresh layout per pair, so a wrong answer is always
a wrong answer *somewhere else on a different die*.  Real navigation error is
harder: the tool is on the correct die and the localiser must pick the right
cell among hundreds of near-identical neighbours.  A family isolates exactly
that failure mode -- every reference in a family shares the same periodic
pattern and the same fingerprint field, so the ONLY thing distinguishing the
sites is local process variation and defect placement.

This also gives two measurements that a per-pair dataset cannot give:
  * confusion within a die -- does site 3's reference match site 7's location?
  * repeatability          -- same site, two independent captures (--repeats),
                              i.e. is the localiser stable against noise alone?

Nothing here re-implements the physics: draw_dram_layout, draw_finfet_layout,
process_variation_field and sem_capture are imported from generate_dataset.py,
so both scripts stay in sync by construction.

Usage
-----
    python generate_family_dataset.py --families 4 --sites 8 --out family_ds
    python generate_family_dataset.py --families 2 --sites 6 --repeats 2 \
        --style finfet --out family_finfet --seed 42
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# generate_dataset.py lives at the repo root (one level up from this file in
# src/), not next to this script -- add it to sys.path so the bare import
# below resolves regardless of the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_dataset import (LAYOUTS, process_variation_field, sem_capture)


def build_family(rng, style, search_size, ref_size, scale_ratio,
                 n_sites, repeats, noise_scale, pv_amplitude,
                 min_sep_frac=0.12):
    """One die -> one search capture + n_sites*repeats reference captures."""
    layout_size = 8000
    layout = LAYOUTS[style](rng, size=layout_size)

    if pv_amplitude is None:
        pv_amplitude = float(rng.uniform(0.0, 0.35))
    pv = process_variation_field(rng, layout_size, amplitude=pv_amplitude)
    layout = np.clip(layout * pv, 0, 1.4).astype(np.float32)

    layout_to_search = layout_size / search_size
    ref_span_search = search_size / scale_ratio
    ref_footprint = int(ref_span_search * layout_to_search)

    # ---- one SEARCH capture for the whole family --------------------------
    angle = rng.uniform(-1.5, 1.5)
    scale_jit = rng.uniform(0.97, 1.03)
    M = cv2.getRotationMatrix2D((layout_size / 2, layout_size / 2),
                                angle, scale_jit)
    deg = cv2.warpAffine(layout, M, (layout_size, layout_size),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    search_hi = cv2.resize(deg, (search_size, search_size),
                           interpolation=cv2.INTER_AREA)
    search = sem_capture(search_hi, rng,
                         blur_sigma=rng.uniform(1.2, 1.8),
                         dose=rng.uniform(50, 90) / noise_scale,
                         read_sigma=rng.uniform(0.03, 0.05),
                         edge_gain=rng.uniform(0.20, 0.35))

    # ---- reference sites, spread out so they are not near-duplicates ------
    half = ref_footprint // 2
    margin = half + 50
    min_sep = min_sep_frac * layout_size
    sites = []
    guard = 0
    while len(sites) < n_sites and guard < n_sites * 400:
        guard += 1
        cx = int(rng.integers(margin, layout_size - margin))
        cy = int(rng.integers(margin, layout_size - margin))
        if all(np.hypot(cx - x, cy - y) >= min_sep for x, y in sites):
            sites.append((cx, cy))
    if len(sites) < n_sites:
        print(f"  note: only {len(sites)} sites fit with min separation "
              f"{min_sep:.0f} layout px")

    refs = []
    for k, (cx_l, cy_l) in enumerate(sites):
        crop = layout[cy_l - half:cy_l + half, cx_l - half:cx_l + half]
        interp = (cv2.INTER_AREA if ref_size <= crop.shape[0]
                  else cv2.INTER_CUBIC)
        hi = cv2.resize(crop, (ref_size, ref_size), interpolation=interp)
        pt = M @ np.array([cx_l, cy_l, 1.0])
        gx, gy = pt[0] / layout_to_search, pt[1] / layout_to_search
        for rep in range(repeats):
            # a fresh capture each repeat: same site, independent noise
            img = sem_capture(hi, rng,
                              blur_sigma=rng.uniform(0.8, 1.2),
                              dose=rng.uniform(180, 260) / noise_scale,
                              read_sigma=rng.uniform(0.01, 0.02),
                              edge_gain=rng.uniform(0.25, 0.40))
            refs.append({"site": k, "repeat": rep,
                         "img": np.uint8(img * 255),
                         "true_center_x": round(float(gx), 2),
                         "true_center_y": round(float(gy), 2)})

    meta = {"style": style, "pv_amplitude": round(pv_amplitude, 4),
            "rotation_deg": round(angle, 3), "scale_jitter": round(scale_jit, 4),
            "ref_span_in_search_px": round(ref_span_search, 1),
            "n_sites": len(sites), "repeats": repeats,
            "min_site_separation_search_px": round(min_sep / layout_to_search, 1)}
    return np.uint8(search * 255), refs, meta


def main():
    ap = argparse.ArgumentParser(
        description="Drift-Sense sample-family dataset generator")
    ap.add_argument("--families", type=int, default=4)
    ap.add_argument("--sites", type=int, default=8,
                    help="reference sites cropped from each die")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent captures of each site (repeatability test)")
    ap.add_argument("--out", type=str, default="family_dataset")
    ap.add_argument("--style", type=str, default="mixed",
                    choices=["dram", "finfet", "mixed"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--search-size", type=int, default=1000)
    ap.add_argument("--ref-size", type=int, default=1000)
    ap.add_argument("--scale-ratio", type=float, default=10.0)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--pv-amplitude", type=float, default=None)
    ap.add_argument("--min-sep-frac", type=float, default=0.12,
                    help="minimum site separation as a fraction of the die")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    records = []

    for f in range(a.families):
        style = (a.style if a.style != "mixed"
                 else ("dram" if f % 2 == 0 else "finfet"))
        search, refs, meta = build_family(
            rng, style, a.search_size, a.ref_size, a.scale_ratio,
            a.sites, a.repeats, a.noise_scale, a.pv_amplitude, a.min_sep_frac)

        sname = f"fam{f:03d}_search.png"
        cv2.imwrite(os.path.join(a.out, sname), search)
        for r in refs:
            rname = f"fam{f:03d}_site{r['site']:02d}_r{r['repeat']}_reference.png"
            cv2.imwrite(os.path.join(a.out, rname), r["img"])
            rec = {"family_id": f, "site_id": r["site"], "repeat": r["repeat"],
                   "reference": rname, "search": sname,
                   "true_center_x": r["true_center_x"],
                   "true_center_y": r["true_center_y"]}
            rec.update({k: v for k, v in meta.items() if k != "n_sites"})
            records.append(rec)
        print(f"[{f + 1}/{a.families}] ({style}) {meta['n_sites']} sites "
              f"x {a.repeats} capture(s) -> {len(refs)} references, "
              f"pv={meta['pv_amplitude']}")

    with open(os.path.join(a.out, "ground_truth.json"), "w") as fh:
        json.dump(records, fh, indent=2)
    keys = ["family_id", "site_id", "repeat", "style", "reference", "search",
            "true_center_x", "true_center_y", "ref_span_in_search_px",
            "rotation_deg", "scale_jitter", "pv_amplitude", "repeats",
            "min_site_separation_search_px"]
    with open(os.path.join(a.out, "ground_truth.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(f"\nDone. {a.families} families, {len(records)} reference captures "
          f"written to '{a.out}/' with ground_truth.json / .csv")
    print("Every reference in a family must be located in that family's single "
          "search image -- wrong answers land on a sibling site, not a "
          "different die.")


if __name__ == "__main__":
    main()
