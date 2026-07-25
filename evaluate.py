"""Run localize.py on a dataset dir and score against ground truth."""
import json, subprocess, sys, time
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "demo10"
tol = 5.0
recs = json.load(open(f"{d}/ground_truth.json"))
errs, times = [], []
for r in recs:
    t0 = time.time()
    out = subprocess.run(
        ["python", "localize.py",
         "--reference", f"{d}/{r['reference']}",
         "--search", f"{d}/{r['search']}"],
        capture_output=True, text=True).stdout.split()
    dt = time.time() - t0
    cx, cy = float(out[0]), float(out[1])
    err = np.hypot(cx - r["true_center_x"], cy - r["true_center_y"])
    errs.append(err); times.append(dt)
    flag = "OK " if err <= tol else "FAIL"
    print(f"{flag} pair {r['pair_id']:>3}: pred=({cx:7.2f},{cy:7.2f}) "
          f"true=({r['true_center_x']:7.2f},{r['true_center_y']:7.2f}) "
          f"err={err:6.2f}px  {dt:.2f}s")
errs = np.array(errs)
print(f"\nAccuracy @ {tol}px tolerance: {(errs<=tol).mean()*100:.0f}% "
      f"({(errs<=tol).sum()}/{len(errs)})")
print(f"Median error: {np.median(errs):.2f}px | Mean time/pair: {np.mean(times):.2f}s")
