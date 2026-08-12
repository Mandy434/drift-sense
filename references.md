# References — Drift-Sense Dataset Generator & Localization Pipeline

Every synthetic-data augmentation and noise-model choice in
`generate_dataset.py`, and every design decision in `localize.py`, is
justified below against the published physics/process literature it is
based on. Numbering matches inline `[C#]` comments in the code.

---

## Dataset generator: SEM imaging model

**[C1] Edge brightening (topographic edge contrast)**
SEM secondary-electron (SE) yield rises sharply at topographic edges and
corners because more of the SE escape depth is exposed to the surface,
producing the characteristic bright outlines seen at line/via edges in real
SEM micrographs.
- Goldstein, J. et al., *Scanning Electron Microscopy and X-Ray
  Microanalysis*, 4th ed., Springer, 2018 — SE yield vs. surface tilt/edge
  geometry (edge/tip effect).
- Reimer, L., *Scanning Electron Microscopy: Physics of Image Formation and
  Microanalysis*, 2nd ed., Springer Series in Optical Sciences, 1998 — SE
  escape-depth model and the resulting edge-brightness effect used for
  CD-SEM linewidth metrology.

**[C2] Poisson shot noise (independent per capture)**
Electron counting at the detector is a Poisson process; noise variance
therefore scales with the electron dose (lower dose / lower magnification →
visibly noisier image), and two separate acquisitions of the same site have
statistically independent noise realizations.
- Timischl, F., Date, M., Nemoto, S., "A statistical model of signal-noise
  in scanning electron microscopy," *Scanning*, 34(3), 2012, 137–144 —
  Poisson-dominated SEM signal-noise model used directly for our
  `rng.poisson(img * dose)` step.

**[C3] Gaussian beam/probe blur**
The finite electron-probe diameter and beam-scan/detector transfer function
are well approximated by a Gaussian point-spread function; higher
magnification (smaller field of view, smaller effective probe footprint per
pixel) gives a sharper image, which is why our reference (100x) uses a
smaller blur sigma than the search image (10x).
- Reimer, L., *Scanning Electron Microscopy*, op. cit. — probe/PSF model and
  magnification-dependent resolution.

**[C4] DRAM word-line / bit-line / contact-via array structure**
DRAM cell arrays are built from a periodic grid of horizontal word-lines and
vertical bit-lines with a storage-node contact at each intersection — the
basis of our `draw_dram_layout()` geometry.
- Jacob, B., Ng, S., Wang, D., *Memory Systems: Cache, DRAM, Disk*, Morgan
  Kaufmann, 2007, Ch. 4 (DRAM array organization) — word-line/bit-line/cell
  layout.

**[C5] FinFET fin / gate geometry**
FinFETs are built from dense, parallel, high-aspect-ratio silicon fins
crossed perpendicularly by polysilicon/metal gate lines, with a distinct
"wrap" region where the gate crosses each fin — the basis of our
`draw_finfet_layout()` geometry (periodic fins + periodic gate rows +
bright crossings).
- Colinge, J.-P. (ed.), *FinFETs and Other Multi-Gate Transistors*, Springer
  Series in Advanced Microelectronics, 2008 — fin/gate array geometry and
  multi-fin standard-cell layout conventions.

## Dataset generator: process-variation fingerprint

**[C6] Across-die CD (critical dimension) / etch non-uniformity**
Real lithography and etch processes exhibit smooth, spatially-correlated
variation in linewidth and feature intensity across a die/wafer (systematic
CD non-uniformity), rather than every unit cell being pixel-identical. Our
`process_variation_field()` models this as a smooth low-frequency
multiplicative modulation applied to the periodic layout before rendering —
this is what makes a given neighborhood in an otherwise-periodic array
locally distinguishable, and is the physical basis for why navigation-error
recovery is solvable at all in a periodic structure.
- Cain, J.P., Spanos, C.J., "Electrical linewidth metrology for systematic
  CD variation characterization and causal analysis," *Proc. SPIE
  Metrology, Inspection, and Process Control for Microlithography*, 2003 —
  systematic across-die/across-wafer CD variation characterization.
- Preil, M.E. et al., "Improving critical dimension uniformity for advanced
  lithography," *Proc. SPIE*, 2001 — spatial CD-uniformity maps and their
  correlation length.

**Per-feature CD / overlay jitter and rare defects (via open/short, fin
break)**
Line-edge roughness, contact/via CD variation, and rare open/short defects
are standard, well-characterized sources of cell-to-cell variation in
periodic semiconductor arrays; they are additional, independent
justification (beyond [C6]) for why the array is periodic but not
identical cell-to-cell.
- Villarrubia, J.S. et al., "Linewidth measurement and standards for
  semiconductor process qualification," *NIST Special Publication*
  (SEMATECH/NIST CD-metrology program) — line-edge roughness and CD
  variation characterization methodology.

---

## Bonus modality: optical microscope (RGB)

The optical path in `generate_dataset.py --modality optical` models four
physical effects. These are standard textbook optics rather than anything
specialised, so the sources below are the canonical references for each.

**[C7] Thin-film interference colour**
A patterned die is a stack of dielectric films over silicon. Reflectance at
each wavelength follows the multi-layer interference condition, so across-die
film *thickness* non-uniformity [C6] appears in a colour capture as a *hue*
shift rather than a brightness shift. This is why the localiser correlates all
three channels jointly instead of converting to luminance first — flattening
to grey discards the chroma-carried fingerprint.
- Macleod, H.A., *Thin-Film Optical Filters*, 4th ed., CRC Press, 2010 —
  multi-layer reflectance and the thickness-dependence of reflected colour.
- Hecht, E., *Optics*, 5th ed., Pearson, 2017, Ch. 9 (Interference) —
  thin-film interference and the reflectance condition behind our per-channel
  `cos(4*pi*n*t/lambda_c)` term.

**[C8] Diffraction-limited imaging and the Airy point-spread function**
An optical objective is diffraction-limited, not probe-limited as in SEM. The
image of a point is an Airy pattern whose radius scales as `lambda / NA`; we
approximate it by a Gaussian with `sigma ~ 0.21 * lambda / NA`. This is why
the 36–96 nm array is *correctly invisible* in optical mode at ~180 nm
resolution (NA 1.40), and why the optical reference field has to be a third of
the search field rather than a tenth.
- Born, M., Wolf, E., *Principles of Optics*, 7th (expanded) ed., Cambridge
  University Press, 1999, Ch. 8–9 — Airy diffraction pattern, Rayleigh
  resolution criterion, and the `lambda / (2 NA)` limit.
- Goodman, J.W., *Introduction to Fourier Optics*, 3rd ed., Roberts & Company Publishers, 2005
  — incoherent imaging as convolution with an intensity PSF, which is the form
  our per-channel blur takes.

**[C9] Lateral chromatic aberration**
The three colour channels are not imaged at identical magnification, so a
feature is displaced slightly differently in R, G and B. We model this as a
small per-channel radial scale difference.
- Hecht, E., *Optics*, op. cit., Ch. 6 (More on Geometrical Optics) — lateral
  (transverse) chromatic aberration and its radial dependence.

**[C10] Bayer colour-filter array and demosaic correlation**
A single-sensor colour camera samples one colour per pixel behind a mosaic
filter and interpolates the rest, which correlates noise between neighbouring
pixels and between channels. Our optical noise is therefore not independent
per channel, as it would be on a three-sensor camera.
- Bayer, B.E., "Color imaging array," US Patent 3,971,065, 1976 — the
  colour-filter mosaic whose interpolation produces the inter-channel
  correlation we model.

---

## Localization pipeline design choices

**Coarse-to-fine multi-scale/rotation NCC search**
Normalized cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) is
the standard classical template-matching similarity measure, invariant to
linear brightness/contrast offset — appropriate given the independent
gain/offset drift in our SEM capture model. A coarse-to-fine grid over scale
(±12% around the nominal 10x ratio) and rotation (±2°) accounts for the
sub-percent stage/optics scale error and small rotation misalignment
between the two captures, which the problem statement identifies as a real
source of navigation error.
- Brunelli, R., *Template Matching Techniques in Computer Vision: Theory and
  Practice*, Wiley, 2009 — NCC formulation and multi-scale/rotation search
  strategy for template matching.

**Periodicity-aware candidate handling + nearest-to-center rule**
In a highly periodic array, NCC produces many near-equal peaks (one per
lattice period) rather than one unambiguous global maximum. Extracting all
peaks within a small margin of the best score, then applying the
challenge's own disambiguation rule (choose the candidate closest to the
center of the search image), directly implements the specification rather
than relying on classical template matching's single-max assumption, which
is exactly what makes plain template matching fail on periodic layouts.

**Channel-count-driven scale ratio and joint colour correlation**
The nominal magnification ratio is read from the channel count: 10x for a
single-channel SEM pair, 3x for a three-channel optical pair (see [C8] for why
the optical field cannot be a tenth). Colour pairs are correlated across all
three channels jointly rather than converted to luminance, because on an
optical capture the film-thickness fingerprint lives largely in hue [C7].

**High-confidence fingerprint override (process-variation verification)**
Because the process-variation field [C6] is a smooth, low-frequency
modulation riding on top of the periodic lattice signal, low-pass filtering
below the array pitch isolates it from the (high-frequency) periodic
structure. Re-scoring ambiguous candidates against this isolated fingerprint
lets the pipeline break ties the pure-lattice NCC cannot — but only when the
fingerprint match is itself strong and the lattice-NCC cost of overriding is
small, so a confident lattice-only win is never demoted by noise in the
low-frequency channel.

**Sub-pixel parabolic refinement**
A quadratic fit to the NCC surface around the integer-pixel peak is a
standard, low-cost way to recover sub-pixel localization accuracy from a
correlation surface without re-running the search at finer resolution.
- Lewis, J.P., "Fast normalized cross-correlation," *Vision Interface*,
  1995 — NCC computation and sub-pixel peak interpolation.

---

*Note: all citations above describe the general physical/statistical
phenomena and classical techniques used to design this pipeline; exact
publication details (volume/page numbers and editions where abbreviated)
should be verified against the publisher's record before final submission if
your institution requires strict citation formatting.*
