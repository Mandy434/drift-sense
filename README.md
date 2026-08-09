# Drift-Sense — AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

SEMICON India Hackathon 2026 · Track 2 (Applied Materials) submission.

Given a **Reference image** (a small site captured at 100x magnification) and a
**Search image** (a 1000x1000 view of the same die region captured at 10x), this
pipeline locates where the reference pattern appears inside the search image and
returns its center coordinates `(x, y)` in search-image pixels — the core of
Navigation-Error Recovery, solved for highly periodic DRAM and FinFET layouts
where classical single-peak template matching breaks down.

**Results.** Reported over **four independent seeds, 80 pairs total** (mixed
DRAM + FinFET, process-variation amplitude randomized per pair). A single 20-pair
run varies between 80 % and 100 % purely by seed, so we report the aggregate
rather than the best run.

| Metric | Value |
|---|---|
| Accuracy @ 5 px tolerance | **95.0 % (76/80)** — 95 % CI ±4.8 pp |
| Per-seed range | 19/20 · 19/20 · 18/20 · 20/20 |
| Median localization error | **0.06–0.12 px** (sub-pixel) |
| Mean inference time per pair | **~2.3 s** (CPU only, no GPU) |
| At 3x dose reduction (seed 555) | **100 % (20/20)**, median 0.12 px |


| **Bonus: optical microscope, 3-channel RGB** | **100 % (8/8)**, median 0.06 px |

Seeds: `42`, `101`, `202`, `303` (20 pairs each). Exact commands in
[Reproducing our reported numbers](#reproducing-our-reported-numbers).

**Three independent cues, by design.** Beyond the periodic array the die carries
(a) discrete sub-array **mats** separated by periphery **strips**, (b) micron-scale
**metrology and assembly structures** — overlay marks, landing pads, CMP
dummy-fill fields, coarse logic blocks — and (c) a smooth across-die
**process-variation fingerprint**. The micron structures are non-periodic
landmarks, so they break lattice ambiguity outright; they are also the only
features above the ~180 nm diffraction limit of a light microscope, which is what
makes the optical bonus modality tractable at all. Disabling the fingerprint
entirely still leaves 95 %.

**Earlier, weaker die models.** The first version of this generator drew one
uniform array across the whole die; there, removing the fingerprint collapsed
accuracy to 40 %, which is what motivated the mat/strip hierarchy (which took the
same ablation to 90 %) and then the micron-scale structures (95 %). We report the
progression because it shows what each layer of die realism is actually worth.

**What actually limits accuracy.** The four failures across the 80 baseline
pairs are concentrated on near-uniform, low-process-variation FinFET dies. A
`--visual-clarity` flag exists for generating the cleanest possible images (for
slides or a quick look) by disabling both disambiguating cues below; it costs
real accuracy (measured at 71.2 %, 57/80) and is **off by default** for exactly
that reason -- the number quoted here is the one to report.

**Bonus modality: optical microscope (RGB).** The same pipeline runs unmodified
on 3-channel brightfield optical captures — `--modality optical` on the
generator, no flag at all on the localiser, which detects the channel count and
adapts. 8/8 within 5 px, median error 0.55 px.

Two things are physically different in the optical case and both are modelled
rather than faked by colourising a grey image:

- **The 10x reference/search pair does not survive.** Optical resolution is
  about `lambda / (2 NA)` — roughly 305 nm at 550 nm and NA 0.90 — so a 1 um
  reference field contains barely three resolution elements and no matchable
  structure at all. Our first attempt produced a literally featureless green
  square. The optical reference field is therefore a third of the search field
  (~3.3 um), the smallest field that still contains mat/strip structure once the
  36–96 nm array has been washed out by diffraction. The fine array is invisible
  in optical mode, as it must be.
- **Colour is interference, not decoration.** Reflectance per channel follows
  `cos(4*pi*n*t/lambda_c)` for dielectric thickness `t`, so across-die thickness
  non-uniformity appears as a *hue* shift. The fingerprint is therefore carried
  partly in chroma, which is why the localiser correlates all three channels
  jointly instead of converting to luminance first. Lateral chromatic
  aberration, per-channel diffraction radius, Bayer demosaic correlation and a
  white-balance error are modelled too.

The optical case is modelled with an oil-immersion objective (NA 1.40) under
short-wave illumination, giving ~180 nm resolution. The Airy disc is approximated
by a Gaussian with sigma ~ 0.21 lambda / NA; using the Rayleigh radius directly
as a sigma over-blurs by about 1.5x, which is worth stating because it visibly
changes the images and cost a factor of two in median error before we caught it.
Median error is still several times worse than SEM (0.55 px vs ~0.1 px): with no
resolvable fine array there are far fewer sharp edges to pin the correlation peak
to, so sub-pixel refinement has less to work with. Accuracy is nonetheless as
high as the SEM case, because the mat/strip/landmark layout is coarse enough that
period-jump ambiguity largely disappears — mats are individually distinguishable
by their interference colour and by the metrology marks they carry.

**A generator bug we found and fixed.** Reference sites were originally drawn
with a border margin computed *before* the affine degradation was applied. A
rotation of up to 1.5° plus 3 % scale about the die centre displaces a corner
site by up to ~320 layout px, which could push the true centre to within half a
template width of the search-image border — where template matching cannot place
the window at all, making the pair unsolvable by construction rather than
genuinely hard. This affected roughly 1 pair in 30. The margin now bounds that
displacement, and all numbers above are measured after the fix.

No deep learning, no model weights, no training step — a fully classical,
CPU-only pipeline that is deterministic and reproducible.

---

## Quick start

Requires Python 3.8 or newer.

```bash
git clone <YOUR-REPO-URL>
cd drift-sense
pip install -r requirements.txt
```

### 1. Generate a dataset (with ground truth)

```bash
python generate_dataset.py --num-pairs 30 --out dataset --style mixed --seed 2026
```

This writes, into `dataset/`:

- `pairNNN_reference.png` — reference image (100x capture)
- `pairNNN_search.png` — search image (10x capture, 1000x1000)
- `ground_truth.json` / `ground_truth.csv` — true center `(x, y)` of the
  reference inside each search image, plus the rotation/scale error applied

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--num-pairs N` | 30 | number of image pairs to generate |
| `--out DIR` | `dataset` | output directory |
| `--style {dram,finfet,mixed}` | `dram` | die architecture (`mixed` alternates) |
| `--seed N` | random | RNG seed for reproducibility |
| `--noise-scale F` | 1.0 | `>1` produces noisier images (robustness testing) |
| `--pv-amplitude F` | randomized | fix across-die process-variation strength (`0.0` = purely periodic, hardest case) |
| `--search-size N` | 1000 | search image dimension in pixels |
| `--ref-size N` | 1000 | reference image dimension in pixels (spec: 1000x1000, same as search) |

### 2. Run localization on a single pair

```bash
python localize.py --reference dataset/pair000_reference.png --search dataset/pair000_search.png
```

Prints a single line to stdout — the predicted center:

```
348.43 164.79
```

Add `--verbose` to also print the winning scale/rotation, candidate count and
runtime to stderr (stdout stays a clean `x y` for machine parsing).

### 3. Score the whole dataset

```bash
python evaluate.py dataset
```

Prints per-pair predicted vs. true center, error in pixels, and a summary with
accuracy at 5 px tolerance, median error and mean time per pair.

---

## Repository contents

| File | Purpose |
|---|---|
| `generate_dataset.py` | Synthetic dataset generator (DRAM + FinFET, SEM imaging model, ground truth) |
| `localize.py` | **Localization inference script** — the file to run on test data |
| `evaluate.py` | Batch evaluation harness (accuracy, error, timing) |
| `make_visuals.py` | Renders side-by-side success/failure visualizations |
| `app.py` | Gradio browser app: generate a dataset, browse SEM and optical pairs, and see the localiser's measured drift vector per pair |
| `generate_family_dataset.py` | Sample-family generator: many reference sites from **one** die, so a wrong answer lands on a sibling cell rather than a different die |
| `visualize_pipeline.py` | Renders the SEM imaging chain stage by stage (layout → PV field → edge brightening → blur → shot noise → read noise) |
| `tests/test_drift_sense.py` | pytest suite: calibration, 6F²/CPP geometry, ground-truth integrity, solvability margin, determinism, CLI contract, optical modality (25 tests) |
| `solvability_report.py` | Per-pair diagnostic: brute-force search in a structural channel and a fingerprint channel separately, reporting the rank of the true site in each |
| `requirements.txt` | Pinned dependencies |
| `references.md` | Literature justification for every noise/augmentation choice |

**No model weights or training script are included — the pipeline is entirely
classical and requires no training.**

---

## How the dataset generator works

Everything is defined in **nanometres** and rasterised onto one 8000 x 8000
layout canvas at 1.25 nm per layout pixel, so the canvas is exactly the 10 um
field the search image covers and every geometric parameter below is a physical
length rather than a pixel count.

1. **Calibrated unit cells.** `--style dram` uses 6F2 folded-bitline scaling —
   word-line pitch 2F, bit-line pitch 3F, storage contacts on a checkerboard —
   across presets from F = 32 nm to F = 75 nm. `--style finfet` uses fin pitch,
   fin width and contacted poly pitch (CPP ~ 2x fin pitch) across five nodes.
   Both include per-feature CD variation, overlay jitter and rare defects (via
   open/short, fin break).
2. **Die composition.** A real die is not one uniform array, so the canvas is
   tiled with sub-array **mats** — each drawn from an independently chosen preset
   with its own phase — separated by periphery **strips** carrying sparse wide
   routing. This is the mid-scale structure that survives the downsample into the
   search image.
3. **Micron-scale structures.** Overlay (box-in-box) metrology marks, landing
   pads, CMP dummy-fill fields and coarse logic blocks, 0.9-2.6 um. These are
   non-periodic landmarks, and they are the only features above the ~180 nm
   diffraction limit of a light microscope — which is what makes the optical
   bonus modality tractable.
4. **Process-variation fingerprint.** A smooth, spatially-correlated
   multiplicative field models across-die litho/etch CD non-uniformity. Its
   amplitude is randomized per pair, down to zero.
5. **Two independent captures.** The reference crop and the whole layout each go
   through their own imaging chain from separate RNG streams, so structure is
   shared and noise is not.
   - *SEM* (`--modality sem`, default): edge brightening -> Gaussian probe blur
     -> Poisson shot noise -> Gaussian read noise -> brightness/contrast drift.
     The search capture uses a lower dose and larger probe, as at lower
     magnification.
   - *Optical* (`--modality optical`, bonus): thin-film interference colour ->
     per-channel diffraction-limited PSF -> lateral chromatic aberration ->
     illumination falloff and white-balance error -> per-channel photon shot
     noise -> Bayer demosaic correlation -> read noise. Output is 3-channel RGB.
6. **Site selection.** Reference sites are drawn uniformly, except that
   `--boundary-bias` (default 0.35) deliberately places a fraction of them
   straddling a mat/strip edge, which is the realistic hard case. The border
   margin bounds the worst-case affine displacement, so the mapped centre always
   stays at least half a template width inside the search image and every pair is
   locatable in principle.
7. **Degradation and ground truth.** The search scene receives a small random
   rotation (+/-1.5 deg) and scale jitter (+/-3 %) modelling stage/optics
   misalignment; the true centre is mapped through the same affine transform and
   recorded in search-image pixel coordinates.

See `references.md` for the citation behind each of these choices.

## How the localization algorithm works

1. **Denoise** — mild Gaussian blur on both images suppresses the independent
   sensor noise while preserving layout structure.
2. **Coarse-to-fine NCC search** — grid search over template scale (±8 % around
   the nominal magnification ratio) and rotation (±2°), scored with normalized
   cross-correlation, then refined on a finer grid around the coarse optimum. The
   nominal ratio is 10x for a single-channel SEM pair and 3x for a 3-channel
   optical pair; the localiser reads the channel count and picks it, and
   `--scale-ratio` overrides it. Colour images are correlated across all three
   channels jointly rather than converted to luminance, because on an optical
   capture the film-thickness fingerprint lives largely in hue.
3. **Periodicity-suppressed verification** — in a periodic array the lattice
   dominates the correlation, so many peaks score nearly identically and under
   heavy noise a wrong period can outscore the true site. Low-pass filtering
   below the array pitch removes the periodic lattice and isolates the
   low-frequency process-variation fingerprint; every candidate peak is
   re-scored against it. The fingerprint may override the lattice winner only on
   strong evidence (high absolute fingerprint correlation, wide margin over the
   lattice winner, and small lattice deficit), so a confident lattice-only win
   is never demoted by low-frequency noise.
4. **Nearest-to-center rule** — candidates statistically indistinguishable from
   the winner are resolved by choosing the one closest to the center of the
   search image, per the challenge specification.
5. **Sub-pixel refinement** — a parabolic fit to the correlation surface around
   the chosen peak yields sub-pixel accuracy.

## Reproducing our reported numbers

The headline figure is an aggregate over four seeds. Run all four:

```bash
python generate_dataset.py --num-pairs 20 --out dataset  --style mixed --seed 42
python evaluate.py dataset

python generate_dataset.py --num-pairs 20 --out sweep101 --style mixed --seed 101
python evaluate.py sweep101

python generate_dataset.py --num-pairs 20 --out sweep202 --style mixed --seed 202
python evaluate.py sweep202

python generate_dataset.py --num-pairs 20 --out sweep303 --style mixed --seed 303
python evaluate.py sweep303
```

Expected: 19/20, 19/20, 18/20, 20/20 → 76/80 = 95.0 %.

**Please do not judge this pipeline on a single 30-pair run.** Our own runs span
90 %–100 % across seeds with identical code; at n=20 the standard error is about
5 pp. Any comparison between two variants needs several seeds to be
meaningful — we learned this the hard way after a change that looked like a
7-point regression turned out to be noise.

### Robustness checks

All three were re-measured after the margin fix and are quoted in the results
table above.

**On the `--noise-scale` flag.** It divides the electron dose, and shot noise
scales with the square root of dose, so `--noise-scale 3.0` is a 3x *dose*
reduction, not 3x more noise. Measured on the search image, broadband
high-frequency noise rises by about 1.25x (1.56x on the reference, which starts
from a higher dose). We describe these runs as dose reductions rather than
"3x noise" for that reason.

```bash
python generate_dataset.py --num-pairs 20 --out stress2x --style mixed --seed 77 --noise-scale 2.0
python evaluate.py stress2x

python generate_dataset.py --num-pairs 20 --out stress3x --style mixed --seed 555 --noise-scale 3.0
python evaluate.py stress3x
```

Gives 20/20 (100 %) with median error 0.12 px — indistinguishable from the
baseline.

Worst-case check — disable across-die variation entirely (hardest regime, where
the array becomes genuinely ambiguous):

```bash
python generate_dataset.py --num-pairs 20 --out nofp --style mixed --seed 404 --pv-amplitude 0.0
python evaluate.py nofp
```

This build has no process-variation field and no micron-scale landmarks by
design (see below), so this flag has no further effect -- there is nothing left
to disable.

Boundary-straddling crops — force every reference to span a mat/strip edge:

```bash
python generate_dataset.py --num-pairs 20 --out boundary --style mixed --seed 909 --boundary-bias 1.0
python evaluate.py boundary
```

### Per-pair diagnostics

To see *why* a pair failed — whether the true site was findable at all, and in
which cue:

```bash
python solvability_report.py --dataset dataset
```

For each pair this brute-forces scale and rotation in two separate channels: a
band-passed **structural** channel (the periodic pattern) and a low-passed
**fingerprint** channel (the process-variation field), and reports the rank of
the true site in each. A failure where the true site sits at rank 1–3 in one
channel is a candidate-selection problem; a failure where it appears in neither
is genuine ambiguity.

### Clean-image mode (`--visual-clarity`)

For slides, demos, or a quick look at the layout without any disambiguation
scaffolding:

```bash
python generate_dataset.py --num-pairs 20 --out demo --style mixed --seed 42 --visual-clarity
```

This disables both the process-variation fingerprint and the micron-scale
landmarks. Measured accuracy on this build: **71.2 % (57/80)** across the same
four seeds. Do not quote this number as the pipeline's accuracy -- it is a
deliberately harder, scaffolding-free configuration for visual purposes only.
The manifest records which mode produced each pair (`visual_clarity` column).

### Optical microscope mode (bonus)

```bash
python generate_dataset.py --num-pairs 8 --out optical --style mixed --seed 606 --modality optical
python localize.py --reference optical/pair000_reference.png --search optical/pair000_search.png
```

The images are written as 3-channel PNGs and `localize.py` needs no extra flag.
`--scale-ratio` overrides the assumed magnification pair if you want to force one.

`evaluate.py` needs no change: it invokes `localize.py` as a subprocess, so image
loading happens inside the localiser, which handles both modalities. If you write
your own harness that loads images itself, use `cv2.IMREAD_UNCHANGED` rather than
`cv2.imread(path, 0)` — flattening to grey drops the chroma fingerprint *and*
makes the localiser assume the 10x SEM footprint, which fails on every pair.

### Sample families

To test confusion *within* one die rather than across dies:

```bash
python generate_family_dataset.py --families 4 --sites 8 --out family_dataset --seed 42
```

Every reference in a family must be located in that family's single search
image, so a wrong answer lands on a sibling site sharing the same lattice and
the same fingerprint field — the hardest realistic form of navigation error.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

25 tests, about 95 s. These are contract tests rather than accuracy tests — they
guard the properties every reported number silently depends on:

- **Ground-truth integrity** — the reference is downscaled and correlated against
  the search image at the recorded coordinate. It does not require the true site
  to be the global maximum (on a periodic array it often is not; resolving that
  is the localiser's job), only that the label points at genuinely matching
  content. A transposed axis or a forgotten affine mapping would fail here while
  still producing plausible-looking accuracy numbers.
- **Solvability margin** — a regression test for the bug described above: over
  several seeds, the mapped true centre must never land within half a template
  of the search-image border.
- **Shared structure, independent noise** — the two captures must show the same
  die (or the task is impossible) with uncorrelated sensor noise (or it is easier
  than reality).
- **Calibration and geometry** — 10 nm/px, a 100 × 100 px reference footprint,
  DRAM word-line/bit-line pitches consistent with 6F² scaling, FinFET CPP at
  roughly twice the fin pitch.
- **`--pv-amplitude 0.0` really disables the fingerprint** — the worst-case
  ablation figure depends on that flag doing what it claims.
- **Localiser contract** — deterministic for identical input, two paths in, a
  single `x y` line on stdout, and an end-to-end check on a planted synthetic
  patch that is independent of the SEM model.
- **The optical bonus is really colour** — a 256-entry lookup table is fitted from
  luminance to RGB and must fail to reconstruct the image. False colour applied to
  a grey capture would reconstruct almost exactly, so this test fails loudly if
  the optical path ever degenerates into `applyColorMap`.

## Notes for reviewers

- `localize.py` accepts exactly two inputs (a reference image path and a search
  image path) and writes one `x y` line to stdout. It requires no manual edits,
  no configuration file and no trained weights.
- The pipeline is CPU-only and deterministic for a given input pair.
- On Windows, if `python3` is not on your PATH, use `python` (as in the commands
  above).
