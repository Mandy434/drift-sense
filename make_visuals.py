import cv2, json, numpy as np, subprocess

def make_visual(d, idx, outname):
    r = json.load(open(f"{d}/ground_truth.json"))[idx]
    ref = cv2.imread(f"{d}/{r['reference']}", 0)
    sea = cv2.imread(f"{d}/{r['search']}", 0)
    out = subprocess.run(["python", "localize.py", "--reference", f"{d}/{r['reference']}",
                          "--search", f"{d}/{r['search']}"], capture_output=True, text=True).stdout.split()
    cx, cy = float(out[0]), float(out[1])
    tx, ty = r["true_center_x"], r["true_center_y"]
    err = np.hypot(cx - tx, cy - ty)

    sea_c = cv2.cvtColor(sea, cv2.COLOR_GRAY2BGR)
    s = 50
    cv2.rectangle(sea_c, (int(tx - s), int(ty - s)), (int(tx + s), int(ty + s)), (0, 220, 0), 2)
    cv2.drawMarker(sea_c, (int(round(cx)), int(round(cy))), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    ref_c = cv2.cvtColor(cv2.resize(ref, (1000, 1000), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cv2.putText(ref_c, "REFERENCE (100x)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(sea_c, f"SEARCH (10x)  green=true  red=predicted  err={err:.2f}px",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    combo = np.hstack([ref_c, np.full((1000, 20, 3), 40, np.uint8), sea_c])
    cv2.imwrite(outname, combo)
    print(f"{outname}: err={err:.2f}px")

# pair 0 = success example (change index if you want a different one)
make_visual("stress3", 0, "success_case.png")

# find the worst pair from your stress3 for the "honest failure" example
import numpy as np
recs = json.load(open("stress3/ground_truth.json"))
errs = []
for i, r in enumerate(recs):
    out = subprocess.run(["python", "localize.py", "--reference", f"stress3/{r['reference']}",
                          "--search", f"stress3/{r['search']}"], capture_output=True, text=True).stdout.split()
    cx, cy = float(out[0]), float(out[1])
    err = np.hypot(cx - r["true_center_x"], cy - r["true_center_y"])
    errs.append(err)
worst_idx = int(np.argmax(errs))
print(f"worst pair = {worst_idx}, err = {errs[worst_idx]:.2f}px")
make_visual("stress3", worst_idx, "failure_case.png")