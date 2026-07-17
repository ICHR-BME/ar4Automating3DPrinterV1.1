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
FIG_SIZE = (3.9, 3.0)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_total_duration(path):
    """
    Return a list of top-level operation durations (seconds) for a single timing CSV.

    Strategy:
      - Depth-0 rows = rows where call_chain does NOT contain '>'.
      - If any depth-0 row has call_chain == 'startup', return [its duration]
        (it is the all-encompassing wrapper used in scan files).
      - Otherwise, return all depth-0 durations individually (one per top-level
        call, e.g. multiple transferPlate entries per file).
    """
    depth0 = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["call_chain"].strip()
            if ">" not in chain:
                depth0.append((chain, float(row["duration_s"])))

    if not depth0:
        return []

    # Prefer the 'startup' wrapper if present
    for name, dur in depth0:
        if name == "startup":
            return [dur]

    # Return all top-level entries as individual measurements
    return [dur for _, dur in depth0]


def load_folder(folder_path):
    """Return list of all durations (s) across all timing CSVs in a folder."""
    files = sorted(glob.glob(os.path.join(folder_path, "timing_*.csv")))
    durations = []
    for p in files:
        durs = load_total_duration(p)
        if durs:
            durations.extend(durs)
        else:
            print(f"  WARNING: no depth-0 entries in {os.path.basename(p)}, skipping.")
    return durations


# ── Load all categories ────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

categories = [
    ("scan2Markers",   "Scan 2\n markers"),
    ("scan3Markers",   "Scan 3\n markers"),
    ("scrapePlate",    "Scrape\nplate"),
    ("doubleTransfer", "Transfer\nplate"),
]

means  = []
stds   = []
counts = []
labels = []

for folder_name, label in categories:
    folder = os.path.join(script_dir, folder_name)
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")
        continue
    durs = load_folder(folder)
    if not durs:
        print(f"WARNING: no valid data in {folder_name}")
        continue
    means.append(np.mean(durs))
    stds.append(np.std(durs))
    counts.append(len(durs))
    labels.append(label)
    print(f"{folder_name}: n={len(durs)}, mean={np.mean(durs):.2f}s, std={np.std(durs):.2f}s")

# ── Bar chart ──────────────────────────────────────────────────────────────────
COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#084594"]

fig, ax = plt.subplots(figsize=FIG_SIZE)

x = np.arange(len(labels))
ax.bar(
    x,
    means,
    yerr=stds,
    color=COLORS[:len(labels)],
    width=0.45,
    capsize=5,
    error_kw=dict(ecolor="k", elinewidth=1.2, capthick=1.2),
    label=[f"{lbl.replace(chr(10), ' ')} (n={n})" for lbl, n in zip(labels, counts)],
)

# Legend with n= counts (bars can't take list labels directly - use patches)
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(facecolor=COLORS[i], label=f"{labels[i].replace(chr(10), ' ')} (n={counts[i]})")
    for i in range(len(labels))
]
ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=2,
    frameon=True,
    fontsize=9,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Duration [s]")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(pad=0.5)
plt.subplots_adjust(bottom=0.38)
out_path = os.path.join(script_dir, "timing_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")

plt.show()
