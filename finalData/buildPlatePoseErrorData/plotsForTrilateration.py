import os
import sys
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt

# ── Import trilateration functions ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trilateration import solve_rigid_body, CORNER_SPACING

# ── Global style (matches analyzeMultipleScanLocations.py) ────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelpad": 2,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
})
FIG_SIZE  = (3.5, 3.0)

# ── Find CSV files ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
all_csvs   = sorted(glob.glob(os.path.join(script_dir, "buildPlatePlacementErrorMeasurements*.csv")))
csv_files  = [p for p in all_csvs if "_results" not in os.path.basename(p)]

if not csv_files:
    raise FileNotFoundError("No buildPlatePlacementErrorMeasurements*.csv files found.")

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_csv(path):
    """
    Read a measurements CSV, return (header, data_rows) where data_rows is a
    list of raw string lists (one per valid row).
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or line[0].strip().startswith("#"):
                continue
            rows.append([col.strip() for col in line])

    if not rows:
        return None, []

    header    = None
    data_rows = rows
    try:
        float(rows[0][0])
    except ValueError:
        header    = rows[0]
        data_rows = rows[1:]

    return header, data_rows


def compute_results(data_rows):
    """
    Run solve_rigid_body on every valid row.
    Returns list of (row, theta_deg, centroid_offset).
    """
    results = []
    for i, row in enumerate(data_rows):
        if len(row) < 4:
            print(f"  WARNING: row {i+1} has fewer than 4 columns, skipping.")
            continue
        try:
            d1      = float(row[0])
            d_cross2 = float(row[1])
            d2      = float(row[2])
            d_cross1 = float(row[3])
        except ValueError as e:
            print(f"  WARNING: row {i+1} could not be parsed ({e}), skipping.")
            continue
        theta_deg, centroid_offset = solve_rigid_body(d1, d2, d_cross1, d_cross2, CORNER_SPACING)
        results.append((row, theta_deg, centroid_offset))
    return results


def print_table(results):
    col_w = 16
    print(f"\n{'Row':<5} {'d1':>{col_w}} {'d_cross2':>{col_w}} {'d2':>{col_w}} "
          f"{'d_cross1':>{col_w}} {'orientation (°)':>{col_w}} {'centroid offset':>{col_w}}")
    print("-" * (5 + (col_w + 1) * 6))
    for i, (row, theta, offset) in enumerate(results):
        print(f"{i+1:<5} {row[0]:>{col_w}} {row[1]:>{col_w}} {row[2]:>{col_w}} {row[3]:>{col_w}} "
              f"{theta:>{col_w}.4f} {offset:>{col_w}.4f}")


def write_results_csv(path, header, results):
    """Write original data + computed columns to a *_results.csv file."""
    base, ext  = os.path.splitext(path)
    out_path   = base + "_results" + ext

    out_rows = []
    if header is not None:
        out_rows.append(header + ["orientation_deg", "centroid_offset"])
    else:
        out_rows.append(["d1", "d_cross2", "d2", "d_cross1", "orientation_deg", "centroid_offset"])

    for row, theta, offset in results:
        out_rows.append(list(row) + [f"{theta:.6f}", f"{offset:.6f}"])

    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)

    print(f"  Results written to: {out_path}")


# ── Process all CSV files ──────────────────────────────────────────────────────
trial_labels      = []
all_pos_vals      = []   # raw centroid offsets from every row of every file
all_ori_vals      = []   # raw orientation angles from every row of every file
all_printer_errs  = []   # raw printer error values from column 4

for idx, path in enumerate(csv_files, start=1):
    label = f"Trial {idx}"
    print(f"\n{'='*60}")
    print(f"  {label}: {os.path.basename(path)}")

    header, data_rows = parse_csv(path)
    if not data_rows:
        print("  WARNING: no valid data rows found, skipping.")
        continue

    results = compute_results(data_rows)
    if not results:
        print("  WARNING: no rows could be computed, skipping.")
        continue

    print_table(results)
    write_results_csv(path, header, results)

    trial_labels.append(label)
    all_pos_vals.extend([r[2] for r in results])
    all_ori_vals.extend([r[1] for r in results])

    # Collect printer error from column index 4 (if present)
    for r in results:
        row = r[0]
        if len(row) >= 5:
            try:
                all_printer_errs.append(float(row[4]))
            except ValueError:
                pass

n_trials = len(trial_labels)
if n_trials == 0:
    raise RuntimeError("No trials could be processed.")

# ── Pool all raw measurements ──────────────────────────────────────────────────
combined_pos_mean = np.mean(all_pos_vals)
combined_pos_std  = np.std(all_pos_vals)
combined_ori_mean = np.mean(all_ori_vals)
combined_ori_std  = np.std(all_ori_vals)
combined_pe_mean  = np.mean(all_printer_errs) if all_printer_errs else 0.0
combined_pe_std   = np.std(all_printer_errs)  if all_printer_errs else 0.0

# ── Single grouped bar plot ────────────────────────────────────────────────────
COLOR_POS = "#2166ac"   # dark blue
COLOR_ORI = "#6baed6"   # lighter blue
COLOR_PE  = "#08519c"   # navy

fig, ax = plt.subplots(figsize=FIG_SIZE)

bar_width = 0.35
x = np.array([0, 1, 2])

bars = ax.bar(
    x,
    [combined_pos_mean, combined_ori_mean, combined_pe_mean],
    yerr=[combined_pos_std, combined_ori_std, combined_pe_std],
    color=[COLOR_POS, COLOR_ORI, COLOR_PE],
    width=bar_width,
    capsize=5,
    error_kw=dict(ecolor="k", elinewidth=1.2, capthick=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels(["Horizontal\nerror [mm]", "Orientation\nerror [°]", "Gap on\nprinter [mm]"])
ax.set_ylabel("Error")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(pad=0.5)
out_path = os.path.join(script_dir, "build_plate_errors_transfer.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")

plt.show()
