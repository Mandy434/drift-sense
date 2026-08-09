#!/usr/bin/env python3
"""
Drift-Sense browser app
=======================
Generate a dataset, browse the pairs, and see what the localiser actually did on
each one.

Two things here are deliberate:

  * The optical gallery runs the generator a second time with
    `--modality optical`, producing real 3-channel brightfield captures — thin-film
    interference colour, per-channel diffraction-limited PSF, chromatic
    aberration. It is NOT a colour map applied to the grey SEM image: false colour
    adds no information (all three channels stay a function of one), so it would
    not demonstrate the RGB bonus.
  * The drift arrow on each pair is drawn from the localiser's prediction to the
    recorded ground truth, with the error printed in pixels. It is a measurement,
    not an illustration — if the prediction is right the arrow is invisibly short,
    which is the honest outcome.

Run:  python app.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJECT_DIR = Path(__file__).parent.resolve()
GEN = PROJECT_DIR / "generate_dataset.py"
LOC = PROJECT_DIR / "localize.py"


# ------------------------------------------------------------------ helpers
def _run(args, cwd=PROJECT_DIR):
    return subprocess.run([sys.executable] + [str(a) for a in args],
                          capture_output=True, text=True, cwd=str(cwd))


def _generate(out_dir, style, pairs, seed, modality):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    p = _run([GEN, "--style", style, "--num-pairs", int(pairs), "--seed", int(seed),
              "--out", out_dir, "--modality", modality])
    if p.returncode != 0:
        return None, p.stderr.strip()[-1500:]
    gt = out_dir / "ground_truth.json"
    if not gt.exists():
        return None, "generator produced no ground_truth.json"
    return json.loads(gt.read_text()), None


def _localize(ref_path, search_path):
    """Call localize.py exactly as a grader would: two paths in, 'x y' out."""
    p = _run([LOC, "--reference", ref_path, "--search", search_path])
    if p.returncode != 0:
        return None
    try:
        a, b = p.stdout.split()[:2]
        return float(a), float(b)
    except ValueError:
        return None


def _annotate(rec, out_dir, run_localiser):
    """Reference beside search, with the measured drift vector if available."""
    ref = cv2.imread(str(out_dir / rec["reference"]), cv2.IMREAD_UNCHANGED)
    sea = cv2.imread(str(out_dir / rec["search"]), cv2.IMREAD_UNCHANGED)
    if ref is None or sea is None:
        return None, None
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    if sea.ndim == 2:
        sea = cv2.cvtColor(sea, cv2.COLOR_GRAY2BGR)

    tx, ty = rec["true_center_x"], rec["true_center_y"]
    span = rec["ref_span_in_search_px"]
    vis = sea.copy()
    h = int(round(span / 2))
    cv2.rectangle(vis, (int(tx) - h, int(ty) - h), (int(tx) + h, int(ty) + h),
                  (0, 255, 0), 2)                                  # ground truth

    caption = f"pair {rec['pair_id']} · {rec['style']} · {rec['modality']}"
    if run_localiser:
        pred = _localize(out_dir / rec["reference"], out_dir / rec["search"])
        if pred is None:
            caption += " · localiser failed"
        else:
            pxs, pys = pred
            err = float(np.hypot(pxs - tx, pys - ty))
            cv2.drawMarker(vis, (int(pxs), int(pys)), (0, 0, 255),
                           cv2.MARKER_CROSS, 26, 2)                # prediction
            if err > 2.0:                     # only draw a vector worth seeing
                cv2.arrowedLine(vis, (int(pxs), int(pys)), (int(tx), int(ty)),
                                (0, 0, 255), 2, tipLength=0.25)
            caption += f" · error {err:.2f} px" + ("  ✓" if err <= 5 else "  ✗")
    else:
        caption += f" · true centre ({tx:.1f}, {ty:.1f})"

    ref_s = cv2.resize(ref, (500, 500), interpolation=cv2.INTER_AREA)
    vis_s = cv2.resize(vis, (500, 500), interpolation=cv2.INTER_AREA)
    pane = np.hstack([ref_s, np.full((500, 8, 3), 255, np.uint8), vis_s])
    return cv2.cvtColor(pane, cv2.COLOR_BGR2RGB), caption


def generate(style, pairs, seed, do_optical, do_localise, progress=gr.Progress()):
    sem_dir = PROJECT_DIR / "app_output_sem"
    opt_dir = PROJECT_DIR / "app_output_optical"

    progress(0.05, desc="generating SEM pairs")
    recs, err = _generate(sem_dir, style, pairs, seed, "sem")
    if err:
        return [], [], None, f"Error generating SEM dataset:\n{err}"

    sem_gallery = []
    for i, rec in enumerate(recs):
        progress(0.1 + 0.5 * i / max(len(recs), 1),
                 desc=f"SEM pair {i + 1}/{len(recs)}"
                      + (" (localising)" if do_localise else ""))
        img, cap = _annotate(rec, sem_dir, do_localise)
        if img is not None:
            sem_gallery.append((img, cap))

    opt_gallery = []
    if do_optical:
        progress(0.65, desc="generating optical pairs")
        orecs, oerr = _generate(opt_dir, style, pairs, seed, "optical")
        if oerr:
            return sem_gallery, [], None, f"Error generating optical dataset:\n{oerr}"
        for i, rec in enumerate(orecs):
            progress(0.7 + 0.28 * i / max(len(orecs), 1),
                     desc=f"optical pair {i + 1}/{len(orecs)}")
            img, cap = _annotate(rec, opt_dir, do_localise)
            if img is not None:
                opt_gallery.append((img, cap))

    msg = [f"{len(sem_gallery)} SEM pairs"]
    if opt_gallery:
        msg.append(f"{len(opt_gallery)} optical (RGB) pairs")
    if do_localise:
        msg.append("green box = ground truth, red cross = localiser prediction, "
                   "arrow = measured error")
    return sem_gallery, opt_gallery, None, " · ".join(msg)


def make_zip():
    zip_path = PROJECT_DIR / "dataset.zip"
    dirs = [d for d in (PROJECT_DIR / "app_output_sem",
                        PROJECT_DIR / "app_output_optical") if d.exists()]
    if not dirs:
        return None
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    fp = os.path.join(root, f)
                    z.write(fp, os.path.join(d.name,
                                             os.path.relpath(fp, d)))
    return str(zip_path)


# ---------------------------------------------------------------------- UI
with gr.Blocks(title="Drift-Sense Generator") as demo:
    gr.Markdown(
        "# Drift-Sense synthetic dataset generator\n"
        "Navigation-error recovery for wafer inspection tools — SEMICON India 2026, "
        "Track 2.\n\n"
        "Each panel shows the **reference** capture on the left and the "
        "**search** image on the right. Enable *Run localiser* to overlay what "
        "`localize.py` predicted against the recorded ground truth."
    )

    with gr.Row():
        style = gr.Dropdown(["dram", "finfet", "mixed"], value="mixed",
                            label="Architecture")
        pairs = gr.Slider(1, 20, value=4, step=1, label="Pairs")
        seed = gr.Number(value=42, label="Seed", precision=0)
    with gr.Row():
        do_optical = gr.Checkbox(value=True,
                                 label="Also generate optical microscope pairs (RGB, bonus)")
        do_localise = gr.Checkbox(value=True,
                                  label="Run localiser and show measured error (~2.5 s/pair)")

    btn = gr.Button("Generate", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("SEM (primary)"):
        sem_gallery = gr.Gallery(label="reference | search", columns=2, height=560)
    with gr.Tab("Optical microscope (RGB, bonus)"):
        opt_gallery = gr.Gallery(label="reference | search", columns=2, height=560)
        gr.Markdown(
            "Real 3-channel brightfield captures: colour comes from thin-film "
            "interference, so across-die thickness variation appears as a hue "
            "shift. Resolution is ~180 nm (NA 1.40), so the 36–96 nm array is "
            "below the diffraction limit and correctly invisible — only mats, "
            "strips and the micron-scale marks and pads resolve. The reference "
            "field is a third of the search field rather than a tenth, because an "
            "optical objective cannot deliver a matchable 1 µm field."
        )

    with gr.Row():
        download_btn = gr.Button("Create ZIP")
        zip_file = gr.File(label="Download")

    btn.click(fn=generate,
              inputs=[style, pairs, seed, do_optical, do_localise],
              outputs=[sem_gallery, opt_gallery, zip_file, status])
    download_btn.click(fn=make_zip, inputs=[], outputs=[zip_file])


if __name__ == "__main__":
    demo.launch()
