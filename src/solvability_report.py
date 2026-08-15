"""
Per-sample solvability check for a Drift-Sense synthetic dataset.

Reads an already-generated dataset (reference/, search/, manifest.csv) and, for
every pair, runs an exhaustive brute-force template search over scale and
rotation.  This is deliberately NOT the production localiser: it is an oracle
that answers one question -- "is the ground-truth site findable in this search
image at all?"

Reported per sample
-------------------
ncc_at_gt    normalised cross-correlation at the true centre (best transform)
best_ncc     the strongest correlation peak anywhere in the search image
err_px       distance from the strongest peak to the true centre
peak_ratio   best peak / strongest competing peak far from it
                 > 1.20  the true site wins clearly
                ~ 1.00  a decoy is just as strong  -> periodicity trap
solvable     err_px <= --tol
difficulty   easy / medium / hard / unsolvable

Why this matters: an accuracy number computed over samples that are physically
unsolvable understates the localiser.  Splitting the two lets you report
"96% on solvable pairs, 4% of pairs excluded as ambiguous by construction"
instead of a single blurred 90%.

Usage
-----
    python check_solvability.py --dataset ./output/train
    python check_solvability.py --dataset ./output/train --write-clean-manifest
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

REF_NM_PER_PX = 1.0
SRC_NM_PER_PX = 10.0
RATIO = SRC_NM_PER_PX / REF_NM_PER_PX      # reference px per search px


# --------------------------------------------------------------- manifest I/O
def find_col(fieldnames, *wanted):
    """Locate a column: exact match first, then substring, longest candidate
    first.  Order matters -- a loose search for "y" would otherwise match a
    column called "style"."""
    low = {f.lower().strip(): f for f in fieldnames}
    for w in wanted:
        if w in low:
            return low[w]
    for w in sorted(wanted, key=len, reverse=True):
        if len(w) < 2:
            continue
        key = w.replace("_", "")
        for f in fieldnames:
            if key in f.lower().replace("_", ""):
                return f
    return None


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"manifest is empty: {path}")
    fn = list(rows[0].keys())

    cx = find_col(fn, "true_center_x", "center_x_px", "centre_x_px",
                  "center_x", "gt_x", "x_px")
    cy = find_col(fn, "true_center_y", "center_y_px", "centre_y_px",
                  "center_y", "gt_y", "y_px")
    rp = find_col(fn, "reference", "reference_path", "ref_path")
    sp = find_col(fn, "search", "search_path", "src_path")
    span = find_col(fn, "ref_span_in_search_px", "ref_span_search", "ref_span")
    if cx is None or cy is None:
        sys.exit(f"could not find ground-truth centre columns in: {fn}")

    print(f"columns            : x={cx}  y={cy}  ref={rp}  search={sp}"
          + (f"  span={span}" if span else ""))
    return rows, cx, cy, rp, sp, fn, span


def resolve_path(base, row, col, subdir, idx):
    """Accept either a path stored in the manifest or the conventional layout."""
    if col and row.get(col):
        p = os.path.join(base, str(row[col]).replace("\\", "/"))
        if os.path.exists(p):
            return p
    cands = [os.path.join(base, subdir, f"{idx:05d}.png"),
             os.path.join(base, subdir, f"{idx:04d}.png"),
             os.path.join(base, subdir, f"{idx}.png"),
             os.path.join(base, f"pair{idx:03d}_{subdir}.png"),
             os.path.join(base, f"{idx:05d}_{subdir}.png")]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ the check
def bandpass(img, lo_sigma, hi_sigma):
    """Remove the smooth illumination / process-variation gradient and the
    highest-frequency noise, leaving the structural band that carries position
    information.  Without this, TM_CCOEFF_NORMED locks onto the low-frequency
    brightness field, which is broad and nearly translation-invariant."""
    f = img.astype(np.float32)
    if lo_sigma > 0:
        f = f - cv2.GaussianBlur(f, (0, 0), lo_sigma)
    if hi_sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), hi_sigma)
    sd = float(f.std())
    return f / sd if sd > 1e-6 else f


def peak_list(res, min_sep, top_n=60):
    """Local maxima of a correlation map, strongest first, min_sep apart."""
    k = int(max(3, 2 * int(min_sep) + 1))
    loc = (res >= cv2.dilate(res, np.ones((k, k), np.uint8)))
    ys, xs = np.nonzero(loc)
    if ys.size == 0:
        return []
    vals = res[ys, xs]
    order = np.argsort(-vals)
    out = []
    for i in order:
        y, x, v = int(ys[i]), int(xs[i]), float(vals[i])
        if all(np.hypot(x - px, y - py) >= min_sep for px, py, _ in out):
            out.append((x, y, v))
        if len(out) >= top_n:
            break
    return out


def rank_of(agg, gt_x, gt_y, half, tol):
    """Rank of the true site among candidate peaks of an aggregate map."""
    peaks = peak_list(agg, min_sep=max(6.0, tol), top_n=120)
    for i, (px_, py_, _) in enumerate(peaks):
        if np.hypot(px_ + half - gt_x, py_ + half - gt_y) <= tol:
            return i + 1, len(peaks)
    return 0, len(peaks)


def check_pair(ref, search, gt_x, gt_y, *, tol, scales, rots, tmpl_px, keep,
               lo_sigma=6.0, hi_sigma=0.8, fp_sigma=6.0, w_struct=0.5):
    """Brute-force scale+rotation search in TWO channels.

    structural  : bandpassed -- the periodic pattern.  Highly repetitive, so it
                  produces dozens of candidates within ~1% of each other.
    fingerprint : low-passed -- the smooth across-die process-variation field.
                  Carries almost no pattern detail but IS locally unique.
    combined    : the mean of the two response maps.

    Reporting all three quantifies what the fingerprint stage is worth: if the
    structural channel puts the true site at rank 20 and the combined channel
    puts it at rank 1, that difference is the fingerprint's contribution."""
    h, w = ref.shape
    c = (w / 2.0, h / 2.0)
    best = {"ncc": -2.0, "x": -1, "y": -1, "scale": 1.0, "rot": 0.0}
    agg_s = agg_f = None          # max response over all transforms, per channel
    srch_s = bandpass(search, lo_sigma, hi_sigma)          # structural band
    srch_f = bandpass(search, 0.0, fp_sigma)               # fingerprint band

    for sc in scales:
        for rt in rots:
            M = cv2.getRotationMatrix2D(c, rt, sc)
            r = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            # keep only the centre fraction: the border is warp fill, not data
            m = int(h * (1 - keep) / 2)
            r = r[m:h - m, m:w - m]
            t = cv2.resize(r, (tmpl_px, tmpl_px), interpolation=cv2.INTER_AREA)
            ts = bandpass(t, lo_sigma, hi_sigma)
            tf = bandpass(t, 0.0, fp_sigma)
            rs = cv2.matchTemplate(srch_s, ts, cv2.TM_CCOEFF_NORMED)
            rf = cv2.matchTemplate(srch_f, tf, cv2.TM_CCOEFF_NORMED)
            # the true site peaks at ITS best transform, which need not be the
            # globally strongest one -- so aggregate, then rank on the aggregate
            agg_s = rs if agg_s is None else np.maximum(agg_s, rs)
            agg_f = rf if agg_f is None else np.maximum(agg_f, rf)

    # Ranks in each channel, computed on the aggregate response maps.
    #
    # The two maps are NOT comparable raw: the low-pass fingerprint map is
    # smooth and sits high almost everywhere, so a raw weighted sum lets it
    # flatten the sharp structural peak.  Standardising each map first makes
    # "how much does this location stand out in its own channel" the quantity
    # being combined, which is what fusion should mean.
    half = tmpl_px / 2.0

    def z(m):
        sd = float(m.std())
        return (m - float(m.mean())) / (sd if sd > 1e-6 else 1.0)

    zs, zf = z(agg_s), z(agg_f)
    agg_c = w_struct * zs + (1 - w_struct) * zf

    def margin(m):
        """How far the top peak stands above the next distinct peak, in sigma.
        Computed WITHOUT the ground truth, so it can be used to pick a channel
        at run time."""
        pk = peak_list(m, min_sep=max(6.0, tol), top_n=3)
        if len(pk) < 2:
            return 0.0
        sd = float(m.std())
        return (pk[0][2] - pk[1][2]) / (sd if sd > 1e-6 else 1.0)
    gx, gy = int(round(gt_x - half)), int(round(gt_y - half))
    inside = 0 <= gy < agg_c.shape[0] and 0 <= gx < agg_c.shape[1]

    ncc_s = float(agg_s[gy, gx]) if inside else -2.0
    ncc_f = float(agg_f[gy, gx]) if inside else -2.0
    rank_s, n_s = rank_of(agg_s, gt_x, gt_y, half, tol)
    rank_f, _ = rank_of(agg_f, gt_x, gt_y, half, tol)
    rank_c, n_c = rank_of(agg_c, gt_x, gt_y, half, tol)
    # "either cue": the better of the two channels, i.e. what a two-stage
    # matcher achieves when it can fall back to whichever cue is informative
    nz = [r for r in (rank_s, rank_f) if r]
    rank_b = min(nz) if nz else 0            # oracle: uses the ground truth
    # adaptive: trust whichever channel is more decisive about its own answer
    m_s, m_f = margin(zs), margin(zf)
    rank_a = rank_s if m_s >= m_f else rank_f
    chan_a = "struct" if m_s >= m_f else "fingerprint"

    _, mx_c, _, loc_c = cv2.minMaxLoc(agg_c)
    bx, by = loc_c[0] + half, loc_c[1] + half
    err = float(np.hypot(bx - gt_x, by - gt_y))
    top1 = err <= tol

    peaks_c = peak_list(agg_c, min_sep=max(6.0, tol), top_n=120)
    second = peaks_c[1][2] if len(peaks_c) > 1 else -2.0
    ratio = float(0.5 * (w_struct * 2 * ncc_s + (1 - w_struct) * 2 * ncc_f)
                  / second) if second > 1e-6 else float("inf")

    rank = rank_a
    if rank == 1:
        diff = "easy"
    elif 2 <= rank <= 5:
        diff = "medium"
    elif 6 <= rank <= 20:
        diff = "hard"
    elif rank > 20:
        diff = "very_hard"
    else:
        diff = "not_in_peaks"

    return {"ncc_struct": round(ncc_s, 4), "ncc_fingerprint": round(ncc_f, 4),
            "ncc_at_gt": round(0.5 * (w_struct * 2 * ncc_s
                                      + (1 - w_struct) * 2 * ncc_f), 4),
            "best_ncc": round(float(mx_c), 4), "err_px": round(err, 2),
            "peak_ratio": round(ratio, 3),
            "rank_struct": rank_s, "rank_fingerprint": rank_f,
            "rank_best": rank_b, "rank_adaptive": rank_a,
            "channel_chosen": chan_a, "margin_struct": round(m_s, 3),
            "margin_fingerprint": round(m_f, 3), "gt_rank": rank_c,
            "n_peaks": n_c, "top1_hit": int(top1), "difficulty": diff}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="folder containing manifest.csv, reference/, search/")
    ap.add_argument("--tol", type=float, default=20.0,
                    help="search-image px within which a peak counts as correct")
    ap.add_argument("--limit", type=int, default=0, help="check only first N pairs")
    ap.add_argument("--tmpl-px", type=int, default=0,
                    help="template size in search px; 0 = derive from ground truth")
    ap.add_argument("--keep", type=float, default=0.70,
                    help="central fraction of the reference used as template")
    ap.add_argument("--scale-range", type=float, default=0.05)
    ap.add_argument("--scale-steps", type=int, default=7)
    ap.add_argument("--rot-range", type=float, default=2.0)
    ap.add_argument("--rot-steps", type=int, default=9)
    ap.add_argument("--lo-sigma", type=float, default=6.0,
                    help="bandpass: sigma of the low-frequency term removed (0=off)")
    ap.add_argument("--hi-sigma", type=float, default=0.8,
                    help="bandpass: mild smoothing to suppress pixel noise")
    ap.add_argument("--write-clean-manifest", action="store_true",
                    help="write ground_truth_solvable.csv keeping pairs whose true "
                         "site is within --keep-rank candidates")
    ap.add_argument("--keep-rank", type=int, default=20,
                    help="rank cut-off used by --write-clean-manifest")
    a = ap.parse_args()

    base = a.dataset
    mpath = None
    for name in ("manifest.csv", "ground_truth.csv", "labels.csv"):
        if os.path.exists(os.path.join(base, name)):
            mpath = os.path.join(base, name)
            break
    if mpath is None:
        sys.exit(f"no manifest.csv / ground_truth.csv in {base}")
    print(f"ground truth       : {os.path.basename(mpath)}")
    rows, cxk, cyk, rpk, spk, fields, spank = load_manifest(mpath)
    if a.limit:
        rows = rows[:a.limit]

    scales = (np.linspace(1 - a.scale_range, 1 + a.scale_range, a.scale_steps)
              if a.scale_steps > 1 else [1.0])
    rots = (np.linspace(-a.rot_range, a.rot_range, a.rot_steps)
            if a.rot_steps > 1 else [0.0])
    # brute force searches the INVERSE of the distortion applied to the reference
    scales = [1.0 / s for s in scales]
    rots = [-r for r in rots]

    out, skipped = [], 0
    for i, row in enumerate(rows):
        rp = resolve_path(base, row, rpk, "reference", i)
        sp = resolve_path(base, row, spk, "search", i)
        if rp is None or sp is None:
            skipped += 1
            continue
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        src = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or src is None:
            skipped += 1
            continue

        gx, gy = float(row[cxk]), float(row[cyk])
        if gx > src.shape[1] * 1.5:            # manifest stored nm, not px
            gx, gy = gx / SRC_NM_PER_PX, gy / SRC_NM_PER_PX

        tp = a.tmpl_px
        if a.tmpl_px <= 0:                     # derive from declared span
            span = float(row[spank]) if spank and row.get(spank) else \
                src.shape[1] / RATIO
            tp = max(24, int(round(span * a.keep)))
        m = check_pair(ref, src, gx, gy, tol=a.tol, scales=scales, rots=rots,
                       tmpl_px=tp, keep=a.keep,
                       lo_sigma=a.lo_sigma, hi_sigma=a.hi_sigma)
        m["id"] = row.get("id", i)
        m["arch"] = row.get("style", row.get("arch", row.get("architecture", "")))
        m["gt_x"], m["gt_y"] = round(gx, 2), round(gy, 2)
        m["pv_amplitude"] = row.get("pv_amplitude", "")
        out.append((row, m))
        f = lambda v: str(v) if v else "-"
        print(f"[{i:04d}] {m['arch']:6s} rank struct={f(m['rank_struct']):>3s} "
              f"fp={f(m['rank_fingerprint']):>3s} comb={f(m['gt_rank']):>3s}"
              f"/{m['n_peaks']:<3d} err={m['err_px']:7.2f}px  {m['difficulty']}")

    if not out:
        sys.exit("no pairs could be read -- check folder layout")

    keys = ["id", "arch", "gt_x", "gt_y", "ncc_struct", "ncc_fingerprint",
            "ncc_at_gt", "best_ncc", "err_px", "peak_ratio", "rank_struct",
            "rank_fingerprint", "gt_rank", "n_peaks", "top1_hit", "difficulty"]
    rpath = os.path.join(base, "solvability.csv")
    with open(rpath, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows([{k: m[k] for k in keys} for _, m in out])

    n = len(out)
    t1 = sum(1 for _, m in out if m["rank_adaptive"] == 1)
    r5 = sum(1 for _, m in out if 1 <= m["rank_adaptive"] <= 5)
    r20 = sum(1 for _, m in out if 1 <= m["rank_adaptive"] <= 20)
    miss = sum(1 for _, m in out if m["rank_adaptive"] == 0)
    print("\n" + "=" * 66)
    print(f"pairs checked        : {n}"
          + (f"  ({skipped} unreadable)" if skipped else ""))
    print(f"tolerance            : {a.tol:g} search px")
    print("--- rank of the true site, by channel ---")
    for lab, key in (("structural (pattern) only", "rank_struct"),
                     ("fingerprint (PV field) only", "rank_fingerprint"),
                     ("adaptive pick (no ground truth)", "rank_adaptive"),
                     ("fixed-weight z fusion", "gt_rank"),
                     ("oracle best of two [upper bound]", "rank_best")):
        a1 = sum(1 for _, m in out if m[key] == 1)
        a5 = sum(1 for _, m in out if 1 <= m[key] <= 5)
        print(f"  {lab:<28}: rank1 {100 * a1 / n:5.1f}%   top5 {100 * a5 / n:5.1f}%")
    ch = {}
    for _, m in out:
        ch[m["channel_chosen"]] = ch.get(m["channel_chosen"], 0) + 1
    print("  channel picked adaptively    : "
          + ", ".join(f"{k} {v}/{n}" for k, v in sorted(ch.items())))
    print("--- how deep a candidate list must be (adaptive channel) ---")
    print(f"true site is rank 1  : {t1:4d}/{n}  ({100 * t1 / n:5.1f}%)"
          "   <- naive top-1 matcher")
    print(f"true site in top 5   : {r5:4d}/{n}  ({100 * r5 / n:5.1f}%)"
          "   <- top-5 + re-ranking")
    print(f"true site in top 20  : {r20:4d}/{n}  ({100 * r20 / n:5.1f}%)")
    print(f"no peak on true site : {miss:4d}/{n}  ({100 * miss / n:5.1f}%)")
    for d in ("easy", "medium", "hard", "very_hard", "not_in_peaks"):
        c = sum(1 for _, m in out if m["difficulty"] == d)
        if c:
            print(f"  {d:<13}: {c:4d}  ({100 * c / n:5.1f}%)")
    for arch in sorted({m["arch"] for _, m in out if m["arch"]}):
        sub = [m for _, m in out if m["arch"] == arch]
        a5 = sum(1 for x in sub if 1 <= x["rank_adaptive"] <= 5)
        print(f"  {arch:<13}: "
              f"{sum(1 for x in sub if x['rank_adaptive'] == 1)}/{len(sub)} at "
              f"rank 1, {a5}/{len(sub)} in top 5, median ncc_at_gt "
              f"{np.median([x['ncc_at_gt'] for x in sub]):.3f}")
    pvs = [(float(m["pv_amplitude"]), m) for _, m in out
           if str(m.get("pv_amplitude", "")).strip() not in ("", "None")]
    if pvs:
        print("--- fingerprint strength vs which channel wins ---")
        for lo, hi, lab in ((0.0, 0.10, "pv < 0.10 (weak)"),
                            (0.10, 0.22, "pv 0.10-0.22"),
                            (0.22, 9.9, "pv > 0.22 (strong)")):
            grp = [m for v, m in pvs if lo <= v < hi]
            if not grp:
                continue
            f1 = 100 * sum(1 for m in grp if m["rank_fingerprint"] == 1) / len(grp)
            s1 = 100 * sum(1 for m in grp if m["rank_struct"] == 1) / len(grp)
            print(f"  {lab:<20} n={len(grp):3d}   struct rank1 {s1:5.1f}%   "
                  f"fingerprint rank1 {f1:5.1f}%")
    print(f"median ncc_at_gt     : "
          f"{np.median([m['ncc_at_gt'] for _, m in out]):.3f}"
          "   (absolute matchability of the true site)")
    print(f"report               : {rpath}")

    if a.write_clean_manifest:
        cpath = os.path.join(base, "ground_truth_solvable.csv")
        with open(cpath, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows([r for r, m in out if m["rank_adaptive"] and
                          m["rank_adaptive"] <= a.keep_rank])
        kept = sum(1 for _, m in out if m["gt_rank"] and m["gt_rank"] <= a.keep_rank)
        print(f"clean manifest       : {cpath}  ({kept} rows)")

    # a low peak_ratio on many pairs is the periodicity trap, not a bug
    b1 = sum(1 for _, m in out if m["rank_best"] == 1)
    s1 = sum(1 for _, m in out if m["rank_struct"] == 1)
    f1 = sum(1 for _, m in out if m["rank_fingerprint"] == 1)
    a1 = t1
    print("\nhow to read this")
    print(f"  structural alone {100 * s1 / n:.1f}% at rank 1, fingerprint alone "
          f"{100 * f1 / n:.1f}%.")
    print(f"  Adaptive pick reaches {100 * a1 / n:.1f}% using no ground truth, "
          f"against an\n  oracle ceiling of {100 * b1 / n:.1f}% -- the gap is "
          f"how often the margin test picks\n  the wrong channel.")
    if b1 > max(s1, f1):
        print("  The oracle beating both single channels means the cues fail on "
              "DIFFERENT\n  pairs: worth keeping both and selecting, not "
              "blending with fixed weights.")
    if not any(str(m.get("pv_amplitude", "")).strip() for _, m in out):
        print("\nwarning: no pv_amplitude column in the ground truth. If this "
              "dataset was\n         generated by an older version of "
              "generate_dataset.py, regenerate it\n         before quoting any "
              "number from this report.")


if __name__ == "__main__":
    main()
