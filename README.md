# Drift-Sense — AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

SEMICON India Hackathon 2026 · Track 2 (Applied Materials) submission.

Given a **Reference image** (a small site captured at 100x magnification) and a
**Search image** (a 1000x1000 view of the same die region captured at 10x), this
pipeline locates where the reference pattern appears inside the search image and
returns its center coordinates `(x, y)` in search-image pixels — the core of
Navigation-Error Recovery, solved for highly periodic DRAM and FinFET layouts
where classical single-peak template matching breaks down.

**Results on our own 30-pair self-evaluation set (mixed DRAM + FinFET, seed 2026,
with process-variation amplitude randomized per pair):**

| Metric | Value |
|---|---|
| Accuracy @ 5 px tolerance | **90 % (27/30)** |
| Median localization error | **0.07 px** (sub-pixel) |
| Mean inference time per pair | **~1.6 s** (CPU only, no GPU) |
| DRAM-style pairs only | 90 % (9/10) |
| FinFET-style pairs only | 90 % (18/20) |
| At 3x training noise (mixed) | 87 % (13/15) |
| Worst case: no across-die variation at all | 60 % (6/10) |

**Known limitation (stated honestly).** Failures occur on both DRAM and
FinFET pairs, typically where the true site and one adjacent lattice period
become statistically close to indistinguishable at low across-die variation —
the low-pass "fingerprint" verification stage (see below) resolves most of
these ties, but occasionally the wrong period scores marginally higher and
the fingerprint's own advantage isn't decisive enough to override it. FinFET
carries an additional structural disadvantage: fins are translation-invariant
along their length, so a small reference window carries fewer independent
localization cues in the y-direction than a DRAM contact-via grid does.
Accuracy also depends on how much across-die CD non-uniformity the die
exhibits: our generator randomizes this per pair
(including near-zero), and in the degenerate case of a perfectly uniform, purely
periodic array the problem becomes genuinely ambiguous and accuracy falls to
~60 %. We report this range rather than the flattering single number a fixed,
strong-variation dataset would produce.

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
| `requirements.txt` | Pinned dependencies |
| `references.md` | Literature justification for every noise/augmentation choice |

**No model weights or training script are included — the pipeline is entirely
classical and requires no training.**

---

## How the dataset generator works

Each pair is produced from one large (8000x8000) procedurally-drawn layout, so
the reference and search captures show the *same* physical structure:

1. **Layout synthesis** — `--style dram` draws periodic horizontal word-lines
   and vertical bit-lines with a contact via at each intersection; `--style
   finfet` draws dense parallel vertical fins crossed by periodic horizontal
   poly gate rows with bright gate-fin crossings. Both include per-feature CD
   (linewidth) variation, overlay position jitter, and rare defects (via
   open/short, fin break).
2. **Process-variation fingerprint** — a smooth, spatially-correlated
   multiplicative field models across-die litho/etch CD non-uniformity. This is
   the physical reason a specific site is distinguishable at all inside an
   otherwise-repeating array, and it survives the 10x downsample.
3. **Two independent SEM captures** — the reference crop and the whole layout
   are each pushed through an independent SEM imaging chain: edge brightening →
   Gaussian probe blur → Poisson shot noise → Gaussian read noise →
   brightness/contrast drift. Noise is drawn from separate RNG streams, never
   reused between the two images. The search capture uses a lower electron dose
   and larger blur, so it is genuinely noisier and softer than the reference —
   as at lower magnification.
4. **Degradation** — the search scene receives a small random rotation (±1.5°)
   and scale jitter (±3 %), modeling stage/optics misalignment between visits.
5. **Ground truth** — the true reference center is mapped through the same
   affine transform and recorded in search-image pixel coordinates.

See `references.md` for the citation behind each of these choices.

## How the localization algorithm works

1. **Denoise** — mild Gaussian blur on both images suppresses the independent
   sensor noise while preserving layout structure.
2. **Coarse-to-fine NCC search** — grid search over template scale (±8 % around
   the nominal 10x ratio) and rotation (±2°), scored with normalized
   cross-correlation, then refined on a finer grid around the coarse optimum.
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

```bash
python generate_dataset.py --num-pairs 30 --out dataset --style mixed --seed 2026
python evaluate.py dataset
```

Robustness checks:

```bash
python generate_dataset.py --num-pairs 10 --out stress2x --style mixed --seed 77 --noise-scale 2.0
python evaluate.py stress2x

python generate_dataset.py --num-pairs 15 --out stress3x --style mixed --seed 555 --noise-scale 3.0
python evaluate.py stress3x
```

Worst-case check — disable across-die variation entirely (hardest regime):

```bash
python generate_dataset.py --num-pairs 10 --out nofp --style mixed --seed 404 --pv-amplitude 0.0
python evaluate.py nofp
```

## Notes for reviewers

- `localize.py` accepts exactly two inputs (a reference image path and a search
  image path) and writes one `x y` line to stdout. It requires no manual edits,
  no configuration file and no trained weights.
- The pipeline is CPU-only and deterministic for a given input pair.
- On Windows, if `python3` is not on your PATH, use `python` (as in the commands
  above).
