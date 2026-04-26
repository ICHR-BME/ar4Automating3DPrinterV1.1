import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelpad": 2,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
})
FIG_SIZE = (3.5, 3.0)

# ── Find CSV files ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files  = sorted(glob.glob(os.path.join(script_dir, "topDistance*.csv")))

if not csv_files:
    raise FileNotFoundError("No topDistance*.csv files found.")

# ── Load and pool all values ───────────────────────────────────────────────────
all_top   = []   # distanceFromTop[mm]
all_side  = []   # distanceFromSide[mm]
all_gap   = []   # gapOnPrinter[mm]

for idx, path in enumerate(csv_files, start=1):
    print(f"\nTrial {idx}: {os.path.basename(path)}")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = [line for line in reader if line and not line[0].strip().startswith("#")]

    if not rows:
        print("  WARNING: empty file, skipping.")
        continue

    # Detect header
    data_rows = rows
    try:
        float(rows[0][0])
    except ValueError:
        data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        row = [c.strip() for c in row]
        if len(row) < 3:
            print(f"  WARNING: row {i+1} has fewer than 3 columns, skipping.")
            continue
        try:
            all_top.append(float(row[0]))
            all_side.append(float(row[1]))
            all_gap.append(float(row[2]))
        except ValueError as e:
            print(f"  WARNING: row {i+1} could not be parsed ({e}), skipping.")

print(f"\nLoaded {len(all_top)} rows across {len(csv_files)} file(s).")

# ── Pool statistics ────────────────────────────────────────────────────────────
top_arr  = np.array(all_top)
side_arr = np.array(all_side)
gap_arr  = np.array(all_gap)

# Deviation of each measurement from the column mean (absolute error from mean)
vert_errs  = np.abs(top_arr  - top_arr.mean())
horiz_errs = np.abs(side_arr - side_arr.mean())

vert_mean  = vert_errs.mean();  vert_std  = vert_errs.std()
horiz_mean = horiz_errs.mean(); horiz_std = horiz_errs.std()
gap_mean   = gap_arr.mean();    gap_std   = gap_arr.std()

# ── Bar chart ──────────────────────────────────────────────────────────────────
COLOR_TOP  = "#2166ac"   # dark blue
COLOR_SIDE = "#6baed6"   # lighter blue
COLOR_GAP  = "#08519c"   # navy

fig, ax = plt.subplots(figsize=FIG_SIZE)

x = np.array([0, 1, 2])
ax.bar(
    x,
    [vert_mean, horiz_mean, gap_mean],
    yerr=[vert_std, horiz_std, gap_std],
    color=[COLOR_TOP, COLOR_SIDE, COLOR_GAP],
    width=0.35,
    capsize=5,
    error_kw=dict(ecolor="k", elinewidth=1.2, capthick=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels(["Vertical\nerror", "Horizontal\nerror", "Gap on\nprinter"])
ax.set_ylabel("Distance [mm]")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(pad=0.5)
out_path = os.path.join(script_dir, "scraping_errors.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")

plt.show()
