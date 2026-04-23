import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_mean(qs: np.ndarray) -> np.ndarray:
    """Hemisphere-consistent quaternion mean (avoids ±180° cancellation)."""
    qs = qs.copy()
    qs[qs @ qs[0] < 0] *= -1
    q = qs.mean(axis=0)
    return q / np.linalg.norm(q)


def quat_angle_from_ref_deg(qs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """SLERP geodesic angle (deg) between each row of *qs* and *q_ref*."""
    dots = np.clip(np.abs(qs @ q_ref), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def quat_consecutive_angle_deg(qs: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between each consecutive pair of quaternions."""
    dots = np.clip(np.abs(np.einsum("ij,ij->i", qs[:-1], qs[1:])), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "markerPoseData")
csv_files  = sorted(glob.glob(os.path.join(data_dir, "scan_raw_measurements*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No scan_raw_measurements*.csv files found in {data_dir}")

frames = []
for idx, path in enumerate(csv_files, start=1):
    df = pd.read_csv(path)
    df["scan_location"] = idx
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

scan_distances = sorted(data["scan_distance"].unique())
scan_locations = sorted(data["scan_location"].unique())
marker_ids     = sorted(data["marker_id"].unique())
movement_ids   = sorted(data["movement_id"].unique())

CAM_POS_COLS  = ["cam_px", "cam_py", "cam_pz"]
CAM_QUAT_COLS = ["cam_qx", "cam_qy", "cam_qz", "cam_qw"]

# ---------------------------------------------------------------------------
# Build per-group (file × marker × distance × movement) time series
# Each group is one continuous sequence of frames at a fixed robot pose.
# ---------------------------------------------------------------------------

MIN_LEN = 10   # discard groups too short to be meaningful

groups = []
for loc in scan_locations:
    for mid in marker_ids:
        for dist in scan_distances:
            for mov in movement_ids:
                sub = data[
                    (data["scan_location"] == loc) &
                    (data["marker_id"]     == mid) &
                    (data["scan_distance"] == dist) &
                    (data["movement_id"]   == mov)
                ]
                if len(sub) < MIN_LEN:
                    continue

                pos  = sub[CAM_POS_COLS].values   # (N, 3)
                quat = sub[CAM_QUAT_COLS].values  # (N, 4)

                q_ref  = quat_mean(quat)
                p_mean = pos.mean(axis=0)

                # Deviations from group mean
                pos_dev_xyz = pos - p_mean                          # (N, 3) metres
                pos_dev_mag = np.linalg.norm(pos_dev_xyz, axis=1)  # (N,)  metres
                ori_dev     = quat_angle_from_ref_deg(quat, q_ref) # (N,)  degrees

                # Frame-to-frame jumps (first difference)
                pos_delta = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # (N-1,) m
                ori_delta = quat_consecutive_angle_deg(quat)               # (N-1,) deg

                groups.append({
                    "scan_location": loc,
                    "marker_id":     mid,
                    "scan_distance": dist,
                    "movement_id":   mov,
                    "n":             len(sub),
                    "pos_dev_xyz":   pos_dev_xyz,
                    "pos_dev_mag":   pos_dev_mag,
                    "ori_dev":       ori_dev,
                    "pos_delta":     pos_delta,
                    "ori_delta":     ori_delta,
                })

# ---------------------------------------------------------------------------
# Colour scheme: 4 distances → 4 distinct colours
# ---------------------------------------------------------------------------

DIST_COLORS = {d: c for d, c in zip(scan_distances,
                                     ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"])}
XYZ_COLORS  = ["tab:red", "tab:green", "tab:blue"]
XYZ_LABELS  = ["ΔX", "ΔY", "ΔZ"]

# ---------------------------------------------------------------------------
# Figure 1 — Example time series per scan distance
#   Upper row: position deviation per axis (mm)
#   Lower row: orientation angular deviation from group mean (degrees)
#   Each column = one scan distance; example taken from file 1, marker 0
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(
    2, len(scan_distances),
    figsize=(4 * len(scan_distances), 7),
    sharey="row",
)

for col, dist in enumerate(scan_distances):
    ex = next(
        (g for g in groups
         if g["scan_location"] == 1
         and g["marker_id"]    == 0
         and g["scan_distance"] == dist),
        None,
    )

    ax_pos = axes1[0, col]
    ax_ori = axes1[1, col]
    ax_pos.set_title(f"{int(round(dist * 100))} cm", fontsize=11)

    if ex is None:
        ax_pos.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_pos.transAxes)
        ax_ori.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_ori.transAxes)
        continue

    fn = np.arange(ex["n"])

    for i, (lbl, c) in enumerate(zip(XYZ_LABELS, XYZ_COLORS)):
        ax_pos.plot(fn, ex["pos_dev_xyz"][:, i] * 1000, color=c,
                    linewidth=0.8, label=lbl)
    ax_pos.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_pos.set_xlabel("Frame index")

    ax_ori.plot(fn, ex["ori_dev"], color=DIST_COLORS[dist], linewidth=0.8)
    ax_ori.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_ori.set_xlabel("Frame index")

    if col == 0:
        ax_pos.set_ylabel("Position deviation (mm)")
        ax_ori.set_ylabel("Orientation deviation (°)")
        ax_pos.legend(loc="upper right", fontsize=7)

fig1.suptitle(
    "Camera-frame Marker Pose Deviation from Group Mean\n"
    "(file 1, marker 0 — one movement segment per scan distance)",
    fontsize=11,
)
plt.tight_layout()
out1 = os.path.join(script_dir, "marker_noise_timeseries.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")

# ---------------------------------------------------------------------------
# Figure 2 — Power Spectral Density (Welch), averaged by scan_distance
#   Left:  position deviation magnitude  [m² / (cycles/frame)]
#   Right: orientation deviation          [deg² / (cycles/frame)]
#   Shaded band = ±1 σ across all groups at that distance
# ---------------------------------------------------------------------------

NPERSEG = 32   # segment length — fixed so all groups share the same freq grid

psd_pos = {d: [] for d in scan_distances}
psd_ori = {d: [] for d in scan_distances}
freqs   = None

for g in groups:
    if g["n"] < NPERSEG:
        continue
    f, Pxx_pos = welch(g["pos_dev_mag"], nperseg=NPERSEG, scaling="density")
    _, Pxx_ori = welch(g["ori_dev"],     nperseg=NPERSEG, scaling="density")
    if freqs is None:
        freqs = f
    psd_pos[g["scan_distance"]].append(Pxx_pos)
    psd_ori[g["scan_distance"]].append(Pxx_ori)

fig2, (ax_pp, ax_po) = plt.subplots(1, 2, figsize=(13, 5))

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]

    if not psd_pos[dist]:
        continue

    arr_pos = np.array(psd_pos[dist])
    arr_ori = np.array(psd_ori[dist])

    for ax, arr, unit in (
        (ax_pp, arr_pos, "m²"),
        (ax_po, arr_ori, "deg²"),
    ):
        mean_ = arr.mean(axis=0)
        std_  = arr.std(axis=0)
        ax.semilogy(freqs, mean_, color=color, linewidth=1.5, label=lbl)
        ax.fill_between(
            freqs,
            np.clip(mean_ - std_, 1e-15, None),
            mean_ + std_,
            color=color, alpha=0.2,
        )

for ax in (ax_pp, ax_po):
    ax.set_xlabel("Normalized frequency (cycles / frame)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Scan distance")

ax_pp.set_ylabel("PSD  (m² / (cycles/frame))")
ax_pp.set_title("Position Deviation PSD")
ax_po.set_ylabel("PSD  (deg² / (cycles/frame))")
ax_po.set_title("Orientation Deviation PSD")

fig2.suptitle(
    f"Power Spectral Density of Camera-frame Marker Pose Noise\n"
    f"(Welch, nperseg={NPERSEG} — mean ± 1σ across all groups and files)",
    fontsize=11,
)
plt.tight_layout()
out2 = os.path.join(script_dir, "marker_noise_psd.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")

# ---------------------------------------------------------------------------
# Figure 3 — Frame-to-frame jump distributions (violin plots)
#   Left:  position jump magnitude (mm)
#   Right: orientation jump magnitude (°)
#   x-axis: scan distance   (all files and markers combined)
# ---------------------------------------------------------------------------

all_pos_deltas = {d: [] for d in scan_distances}
all_ori_deltas = {d: [] for d in scan_distances}

for g in groups:
    all_pos_deltas[g["scan_distance"]].extend(g["pos_delta"].tolist())
    all_ori_deltas[g["scan_distance"]].extend(g["ori_delta"].tolist())

fig3, (ax_dp, ax_do) = plt.subplots(1, 2, figsize=(12, 5))
x_labels = [f"{int(round(d * 100))} cm" for d in scan_distances]
positions = range(len(scan_distances))

for ax, delta_dict, unit, title in (
    (ax_dp, all_pos_deltas, "mm",  "Position Jump Magnitude per Frame"),
    (ax_do, all_ori_deltas, "°",   "Orientation Jump Magnitude per Frame"),
):
    scale = 1000.0 if unit == "mm" else 1.0
    data_list = [np.array(delta_dict[d]) * scale for d in scan_distances]

    vp = ax.violinplot(data_list, positions=positions,
                       showmedians=True, showextrema=True)
    for body, dist in zip(vp["bodies"], scan_distances):
        body.set_facecolor(DIST_COLORS[dist])
        body.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Scan distance")
    ax.set_ylabel(f"Frame-to-frame Δ ({unit})")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate median values
    for xi, dist in zip(positions, scan_distances):
        med = np.median(delta_dict[dist]) * scale
        ax.text(xi, ax.get_ylim()[1] * 0.97, f"med={med:.3g}",
                ha="center", va="top", fontsize=7)

fig3.suptitle(
    "Frame-to-Frame Marker Pose Jump Distributions vs Scan Distance\n"
    "(all files and marker IDs combined)",
    fontsize=11,
)
plt.tight_layout()
out3 = os.path.join(script_dir, "marker_noise_deltas.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out3}")

# ---------------------------------------------------------------------------
# Figure 4 — Orientation noise: before vs after a Butterworth low-pass filter
#
# Left subplot:  mean Welch PSD of orientation deviation, unfiltered vs
#                filtered at a representative cutoff, one curve per scan
#                distance.
# Right subplot: RMS of orientation deviation (raw) and of the filtered
#                signal as a function of cutoff frequency, showing how much
#                noise is retained after filtering.
# ---------------------------------------------------------------------------

CUTOFFS    = np.linspace(0.005, 0.48, 80)  # scipy Wn (0–1 normalized, 1 = Nyquist = 15 Hz at 30 fps)
                                            # 0.005 → 0.075 Hz  |  0.48 → 7.2 Hz
FILT_ORDER = 1        # 1st-order IIR (RC filter), matching ArucoDetector.py
FPS        = 30.0     # camera frame rate used during data collection
FC_DEMO    = 0.02     # ArucoDetector fCutoff=0.3 Hz → Wn = 0.3/(30/2) = 0.02

# Per-group: collect raw PSD, filtered PSD (at FC_DEMO), and RMS curves
psd_ori_raw  = {d: [] for d in scan_distances}
psd_ori_filt = {d: [] for d in scan_distances}
rms_raw      = {d: [] for d in scan_distances}   # scalar per group
rms_filt_curves = {d: [] for d in scan_distances}  # array over CUTOFFS

b_demo, a_demo = butter(FILT_ORDER, FC_DEMO, btype="low")

for g in groups:
    n = g["n"]
    if n < 4 * FILT_ORDER + 1:
        continue

    ori_raw  = g["ori_dev"]
    ori_filt_demo = filtfilt(b_demo, a_demo, ori_raw)

    # Welch PSDs
    f, Praw  = welch(ori_raw,       nperseg=NPERSEG, scaling="density")
    _, Pfilt = welch(ori_filt_demo, nperseg=NPERSEG, scaling="density")
    psd_ori_raw [g["scan_distance"]].append(Praw)
    psd_ori_filt[g["scan_distance"]].append(Pfilt)

    # Unfiltered RMS (constant vs cutoff)
    rms_raw[g["scan_distance"]].append(np.sqrt(np.mean(ori_raw ** 2)))

    # Filtered RMS as function of cutoff
    rms_fc = []
    for fc in CUTOFFS:
        b, a = butter(FILT_ORDER, fc, btype="low")
        rms_fc.append(np.sqrt(np.mean(filtfilt(b, a, ori_raw) ** 2)))
    rms_filt_curves[g["scan_distance"]].append(rms_fc)

fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(13, 5))

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]

    # --- Left: PSD before vs after ---
    if psd_ori_raw[dist]:
        raw_mean  = np.array(psd_ori_raw [dist]).mean(axis=0)
        filt_mean = np.array(psd_ori_filt[dist]).mean(axis=0)
        ax4l.semilogy(f, raw_mean,  color=color, linewidth=1.8,
                      linestyle="-",  label=f"{lbl} raw")
        ax4l.semilogy(f, filt_mean, color=color, linewidth=1.8,
                      linestyle="--", label=f"{lbl} filtered")

    # --- Right: RMS retained vs cutoff ---
    if rms_filt_curves[dist]:
        raw_rms_mean  = np.mean(rms_raw[dist])
        filt_arr      = np.array(rms_filt_curves[dist])    # (n_groups, n_cutoffs)
        filt_mean_    = filt_arr.mean(axis=0)
        filt_std_     = filt_arr.std(axis=0)

        # Unfiltered reference as horizontal line
        ax4r.axhline(raw_rms_mean, color=color, linewidth=1.0,
                     linestyle=":", alpha=0.7)
        ax4r.plot(CUTOFFS, filt_mean_, color=color, linewidth=1.8, label=lbl)
        ax4r.fill_between(CUTOFFS,
                          filt_mean_ - filt_std_,
                          filt_mean_ + filt_std_,
                          color=color, alpha=0.15)

# Mark the demo cutoff on both axes
ax4l.axvline(FC_DEMO, color="k", linewidth=1.0, linestyle="--",
             label=f"ArucoDetector f_c = {FC_DEMO} ({FC_DEMO * FPS / 2:.2g} Hz)")
ax4r.axvline(FC_DEMO, color="k", linewidth=1.0, linestyle="--",
             label=f"ArucoDetector f_c = {FC_DEMO} ({FC_DEMO * FPS / 2:.2g} Hz)")

# Annotate additional reference cutoffs on the right panel
for fc_ann, ls in ((0.01, ":"), (0.05, "-."), (0.1, ":")):
    ax4r.axvline(fc_ann, color="grey", linewidth=0.8, linestyle=ls)
    ax4r.text(fc_ann + 0.002,
              ax4r.get_ylim()[0] if ax4r.get_ylim()[0] > 0 else 0,
              f"{fc_ann}\n({fc_ann * FPS / 2:.2g} Hz)",
              color="grey", fontsize=6, va="bottom")

ax4l.set_xlabel("Normalized frequency (cycles / frame)")
ax4l.set_ylabel("PSD  (deg² / (cycles/frame))")
ax4l.set_title(f"Orientation PSD: Raw vs Low-Pass\n(f_c = {FC_DEMO} → {FC_DEMO * FPS / 2:.2g} Hz at {FPS:.0f} fps)")
ax4l.legend(fontsize=7, ncol=2)
ax4l.grid(True, which="both", alpha=0.3)

ax4r.set_xlabel(f"Low-pass cutoff Wn (0–1 normalized, 1 = Nyquist = {FPS/2:.0f} Hz)")
ax4r.set_ylabel("Orientation RMS (°)")
ax4r.set_title("Orientation RMS Retained After Low-Pass Filter")
ax4r.legend(title="Scan distance  (dotted = unfiltered)", fontsize=8)
ax4r.grid(True, alpha=0.3)

fig4.suptitle(
    f"Low-Pass Filter Effect on Orientation Noise\n"
    f"(Order-{FILT_ORDER} Butterworth matching ArucoDetector RC filter: "
    f"f_c = {FC_DEMO * FPS / 2:.2g} Hz at {FPS:.0f} fps — "
    f"mean ± 1σ across all groups and files)",
    fontsize=11,
)
plt.tight_layout()
out4 = os.path.join(script_dir, "marker_noise_lpf_effect.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved {out4}")

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

print("\nFrame-to-frame position jump statistics (mm):")
print(f"{'Distance':>10}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'95th %':>8}")
for dist in scan_distances:
    arr = np.array(all_pos_deltas[dist]) * 1000
    print(f"{int(round(dist*100)):>8}cm  {arr.mean():>8.4f}  "
          f"{np.median(arr):>8.4f}  {arr.std():>8.4f}  "
          f"{np.percentile(arr, 95):>8.4f}")

print("\nFrame-to-frame orientation jump statistics (°):")
print(f"{'Distance':>10}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'95th %':>8}")
for dist in scan_distances:
    arr = np.array(all_ori_deltas[dist])
    print(f"{int(round(dist*100)):>8}cm  {arr.mean():>8.4f}  "
          f"{np.median(arr):>8.4f}  {arr.std():>8.4f}  "
          f"{np.percentile(arr, 95):>8.4f}")

plt.show()
