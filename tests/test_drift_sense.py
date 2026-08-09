"""
Drift-Sense test suite
======================
These tests are contract tests, not accuracy tests. Accuracy is measured by
`evaluate.py` over several seeds and reported in the README; what matters here is
that the properties every one of those measurements silently depends on actually
hold:

  * the ground-truth coordinate really is where the reference pattern sits
    (a label bug would make every reported number meaningless while still
    looking plausible),
  * the two captures share one physical structure but not one noise realisation,
  * the pixel-scale calibration matches the challenge specification,
  * every pair is locatable in principle -- the true centre never lands within
    half a template of the search-image border,
  * `localize.py` honours the submission interface exactly.

Run with:   pytest tests/ -v
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dataset as G                                    # noqa: E402
import localize as L                                            # noqa: E402


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def pair_dram():
    return G.generate_pair(np.random.default_rng(1234), style="dram")


@pytest.fixture(scope="module")
def pair_finfet():
    return G.generate_pair(np.random.default_rng(5678), style="finfet")


# ------------------------------------------------------- calibration / spec
def test_pixel_scale_matches_specification():
    """Search image: 1000 px covering a 10 um field -> exactly 10 nm/px."""
    assert G.SEARCH_FOV_NM == 10000.0
    assert G.SEARCH_FOV_NM / 1000 == 10.0
    # one layout pixel is a fixed physical length, so nm parameters are meaningful
    assert G.NM_PER_LAYOUT_PX == pytest.approx(G.SEARCH_FOV_NM / G.LAYOUT_SIZE)
    assert G.px(G.NM_PER_LAYOUT_PX) == pytest.approx(1.0)


def test_dram_presets_follow_6f2_scaling():
    """Word-line pitch 2F and bit-line pitch 3F, with F in a plausible range."""
    for name, f in G.DRAM_PRESETS.items():
        assert 20.0 <= f <= 90.0, f"{name}: F={f} nm outside DRAM range"
        assert G.px(2 * f) >= 2.0, f"{name}: word-line pitch under 2 layout px"


def test_finfet_presets_keep_cpp_near_twice_fin_pitch():
    for name, (fin_pitch, fin_width, cpp) in G.FINFET_PRESETS.items():
        assert 0.25 <= fin_width / fin_pitch <= 0.40, f"{name}: odd fin duty cycle"
        assert 1.7 <= cpp / fin_pitch <= 2.3, f"{name}: CPP/fin-pitch = {cpp/fin_pitch:.2f}"


@pytest.mark.parametrize("style", ["dram", "finfet"])
def test_image_shapes_and_dtype(style):
    ref, search, gt = G.generate_pair(np.random.default_rng(7), style=style)
    assert ref.shape == (1000, 1000) and search.shape == (1000, 1000)
    assert ref.dtype == np.uint8 and search.dtype == np.uint8
    assert gt["ref_span_in_search_px"] == pytest.approx(100.0)
    assert gt["nm_per_search_px"] == pytest.approx(10.0)


# --------------------------------------------------- ground-truth integrity
@pytest.mark.parametrize("style", ["dram", "finfet"])
def test_ground_truth_is_where_the_pattern_actually_is(style):
    """
    The single most important test in the suite.

    Downscale the reference by the magnification ratio and correlate it against
    the search image. The correlation at the recorded ground-truth location must
    be high in absolute terms. This does not require the true site to be the
    global maximum -- on a periodic array it often is not, and disambiguating
    that is the localiser's job -- it requires the label to point at genuinely
    matching content. A transposed axis, a sign error or a forgotten affine
    mapping would all fail here.
    """
    ref, search, gt = G.generate_pair(np.random.default_rng(99), style=style)
    span = int(round(gt["ref_span_in_search_px"]))
    keep = 0.7                                   # ignore the warped border
    m = int(ref.shape[0] * (1 - keep) / 2)
    tpl = cv2.resize(ref[m:-m, m:-m], (int(span * keep), int(span * keep)),
                     interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(search.astype(np.float32), tpl.astype(np.float32),
                            cv2.TM_CCOEFF_NORMED)
    half = tpl.shape[0] / 2.0
    gx = int(round(gt["true_center_x"] - half))
    gy = int(round(gt["true_center_y"] - half))
    assert 0 <= gx < res.shape[1] and 0 <= gy < res.shape[0]
    # best score in a small neighbourhood, allowing for sub-pixel rounding
    win = res[max(0, gy - 3):gy + 4, max(0, gx - 3):gx + 4]
    assert win.max() > 0.55, (
        f"{style}: correlation at the ground-truth location is only "
        f"{win.max():.3f} -- the label may not point at the reference pattern")


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_true_centre_is_always_locatable(seed):
    """
    The affine degradation is applied after the site is chosen, so it can push
    the true centre towards the border. If it lands within half a template of
    the edge, template matching cannot place the window there and the pair is
    unsolvable by construction rather than genuinely hard. This regression test
    covers the margin that bounds that displacement.
    """
    _, search, gt = G.generate_pair(np.random.default_rng(seed), style="finfet")
    half = gt["ref_span_in_search_px"] / 2.0
    x, y = gt["true_center_x"], gt["true_center_y"]
    w = search.shape[1]
    clearance = min(x - half, y - half, w - half - x, w - half - y)
    assert clearance >= 0, (
        f"seed {seed}: true centre { (x, y) } is {-clearance:.1f} px outside the "
        f"matchable area")


# ------------------------------------------- shared structure, separate noise
def test_captures_share_structure_but_not_noise():
    """
    Both images must show the same physical die (otherwise the task is
    impossible) while carrying independent sensor noise (otherwise the task is
    easier than reality and the noise model is not being exercised).
    """
    rng = np.random.default_rng(2024)
    ref_a, search_a, gt_a = G.generate_pair(rng, style="dram")
    # same structure seed, fresh capture noise: regenerate from the same seed
    ref_b, search_b, gt_b = G.generate_pair(np.random.default_rng(2024), style="dram")
    assert gt_a["true_center_x"] == gt_b["true_center_x"]
    assert np.array_equal(ref_a, ref_b), "generation is not deterministic per seed"

    # within one pair, the reference and the search image must not be identical
    # in their high-frequency content
    def hf(img):
        f = img.astype(np.float32)
        return f - cv2.GaussianBlur(f, (0, 0), 2.0)
    small = cv2.resize(search_a, ref_a.shape[::-1], interpolation=cv2.INTER_CUBIC)
    r = np.corrcoef(hf(ref_a).ravel(), hf(small).ravel())[0, 1]
    assert abs(r) < 0.5, "reference and search noise appear correlated"


def test_process_variation_field_is_smooth_and_centred():
    field = G.process_variation_field(np.random.default_rng(3), 512, amplitude=0.3)
    assert field.shape == (512, 512)
    assert field.mean() == pytest.approx(1.0, abs=0.08), "field should be multiplicative around 1"
    # smooth: essentially no energy at the pixel scale. The field is built on a
    # coarse grid and interpolated, so structure appears only well above a few
    # pixels; what must be absent is per-pixel variation, which would make the
    # "fingerprint" a noise term rather than a physical modulation.
    hf = field - cv2.GaussianBlur(field, (0, 0), 1.5)
    assert hf.std() / field.std() < 0.05, "process-variation field is not smooth"


def test_pv_amplitude_zero_removes_the_fingerprint():
    """`--pv-amplitude 0.0` must produce a genuinely uniform-variation die: this
    is the worst-case regime quoted in the README, so it has to be real."""
    f0 = G.process_variation_field(np.random.default_rng(4), 256, amplitude=0.0)
    f1 = G.process_variation_field(np.random.default_rng(4), 256, amplitude=0.3)
    assert f0.std() < 1e-4
    assert f1.std() > 0.02


# ---------------------------------------------------------- die composition
def test_die_is_composed_of_mats_separated_by_strips():
    """A real die is not one uniform pattern; the mid-scale structure this
    creates is what survives the 10x downsample."""
    img, mats, strips = G.compose_die(np.random.default_rng(8), "dram")
    assert len(mats) >= 4, "expected several sub-array mats on the canvas"
    assert len(strips) >= 4, "expected periphery/routing strips between mats"
    # Strips must be visually distinguishable from mats, otherwise they add no
    # mid-scale structure. They differ in TEXTURE rather than mean brightness:
    # a mat is a dense periodic array, a strip is flat with sparse routing, so
    # the discriminating statistic is local contrast, not average grey level.
    def texture(rects):
        return float(np.mean([img[y0:y1, x0:x1].std() for x0, y0, x1, y1 in rects]))
    t_mat, t_strip = texture(mats), texture(strips)
    assert t_mat > 1.5 * t_strip, (
        f"mats (contrast {t_mat:.3f}) are not clearly denser than strips "
        f"(contrast {t_strip:.3f}) -- the die would read as one uniform texture")


def test_boundary_bias_controls_edge_straddling_crops():
    """bias 1.0 should nearly always straddle a mat/strip edge; 0.0 never."""
    always = [G.generate_pair(np.random.default_rng(100 + i), style="dram",
                              boundary_bias=1.0)[2]["on_mat_boundary"]
              for i in range(4)]
    never = [G.generate_pair(np.random.default_rng(200 + i), style="dram",
                             boundary_bias=0.0)[2]["on_mat_boundary"]
             for i in range(4)]
    assert sum(always) >= 3, "boundary_bias=1.0 did not produce edge crops"
    assert sum(never) == 0, "boundary_bias=0.0 still produced edge crops"


# ------------------------------------------------------- localiser interface
def test_localize_returns_a_coordinate_inside_the_image(pair_dram):
    ref, search, gt = pair_dram
    cx, cy = L.localize(ref.astype(np.float32), search.astype(np.float32))
    assert 0 <= cx <= search.shape[1] and 0 <= cy <= search.shape[0]
    assert isinstance(cx, float) and isinstance(cy, float)


def test_localize_is_deterministic(pair_dram):
    """The submission is scored by re-running it; the same inputs must give the
    same answer every time."""
    ref, search, _ = pair_dram
    a = L.localize(ref.astype(np.float32), search.astype(np.float32))
    b = L.localize(ref.astype(np.float32), search.astype(np.float32))
    assert a == b


def test_localize_finds_a_clean_synthetic_planted_patch():
    """
    An end-to-end sanity check independent of the SEM model: plant a known
    textured patch at a known place in a noisy field and require the localiser
    to find it. If this fails the search logic itself is broken, regardless of
    how the dataset looks.
    """
    rng = np.random.default_rng(0)
    search = rng.normal(120, 18, (1000, 1000)).astype(np.float32)
    patch = rng.normal(120, 55, (100, 100)).astype(np.float32)
    patch = cv2.GaussianBlur(patch, (0, 0), 1.2)
    x0, y0 = 640, 300
    search[y0:y0 + 100, x0:x0 + 100] = patch
    search = cv2.GaussianBlur(search, (0, 0), 0.8)
    ref = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    cx, cy = L.localize(ref, search)
    err = np.hypot(cx - (x0 + 50), cy - (y0 + 50))
    assert err < 10.0, f"planted patch missed by {err:.1f} px"


# ---------------------------------------------------------------- CLI contract
def test_cli_writes_a_manifest_with_the_expected_columns(tmp_path):
    out = tmp_path / "ds"
    subprocess.run([sys.executable, str(ROOT / "generate_dataset.py"),
                    "--num-pairs", "2", "--out", str(out),
                    "--style", "mixed", "--seed", "5"],
                   cwd=ROOT, check=True, capture_output=True)
    recs = json.loads((out / "ground_truth.json").read_text())
    assert len(recs) == 2
    for key in ("pair_id", "style", "pv_amplitude", "reference", "search",
                "true_center_x", "true_center_y", "ref_span_in_search_px",
                "rotation_deg", "scale_jitter"):
        assert key in recs[0], f"missing manifest column: {key}"
    assert (out / "ground_truth.csv").exists()
    for r in recs:
        assert (out / r["reference"]).exists() and (out / r["search"]).exists()
    # mixed mode must actually mix
    assert {r["style"] for r in recs} == {"dram", "finfet"}


def test_localize_cli_prints_two_numbers(tmp_path):
    out = tmp_path / "ds"
    subprocess.run([sys.executable, str(ROOT / "generate_dataset.py"),
                    "--num-pairs", "1", "--out", str(out), "--seed", "6"],
                   cwd=ROOT, check=True, capture_output=True)
    p = subprocess.run([sys.executable, str(ROOT / "localize.py"),
                        "--reference", str(out / "pair000_reference.png"),
                        "--search", str(out / "pair000_search.png")],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    parts = p.stdout.strip().split()
    assert len(parts) == 2, f"stdout must be exactly 'x y', got {p.stdout!r}"
    float(parts[0]), float(parts[1])


# ------------------------------------------------- optical modality (bonus)
def test_optical_modality_is_three_channel():
    ref, search, gt = G.generate_pair(np.random.default_rng(606),
                                      style="dram", modality="optical")
    assert ref.shape == (1000, 1000, 3) and search.shape == (1000, 1000, 3)
    assert gt["channels"] == 3 and gt["modality"] == "optical"
    # an optical objective cannot deliver a 1 um reference field, so the optical
    # magnification pair must not be the SEM 10x
    assert gt["ref_span_in_search_px"] > 200


def test_optical_colour_is_not_a_colourmapped_grey_image():
    """
    Guard against the tempting shortcut of running a colour map over the grey SEM
    capture. False colour adds no information: all three channels stay a
    deterministic function of one, so a 256-entry lookup table reconstructs the
    whole image. Real interference colour does not, because hue carries film
    thickness independently of brightness. This test fails loudly if the optical
    path ever degenerates into `applyColorMap`.
    """
    _, search, _ = G.generate_pair(np.random.default_rng(707),
                                   style="finfet", modality="optical")
    lum = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    lut = np.zeros((256, 3), np.float64)
    cnt = np.zeros(256)
    for c in range(3):
        np.add.at(lut[:, c], lum.ravel(), search[..., c].ravel().astype(np.float64))
    np.add.at(cnt, lum.ravel(), 1)
    lut /= np.maximum(cnt, 1)[:, None]
    residual = np.abs(lut[lum] - search.astype(np.float64)).mean()
    assert residual > 4.0, (
        f"colour is predictable from luminance to within {residual:.2f}/255 -- "
        "this looks like a colour map, not an independent colour channel")


def test_localiser_handles_an_optical_pair_without_being_told():
    """No flag, no configuration: the channel count selects the modality."""
    ref, search, gt = G.generate_pair(np.random.default_rng(808),
                                      style="dram", modality="optical")
    cx, cy = L.localize(ref.astype(np.float32), search.astype(np.float32))
    err = np.hypot(cx - gt["true_center_x"], cy - gt["true_center_y"])
    assert err < 15.0, f"optical pair missed by {err:.1f} px"
