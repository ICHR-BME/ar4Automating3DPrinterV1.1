import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({"font.size": 12})
FIG_SIZE = (3.5, 3.0)

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
# Build per-group (trial × marker × distance × movement) time series
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
                                     ["#c6dbef", "#6baed6", "#2171b5", "#084594"])}
XYZ_COLORS  = ["#6baed6", "#2171b5", "#084594"]
XYZ_LABELS  = ["ΔX", "ΔY", "ΔZ"]

# ---------------------------------------------------------------------------
# Figure 1a/1b - Example time series per scan distance (separate figures)
#   Position deviation per axis (mm) and orientation deviation (degrees)
#   Example taken from trial 1, marker 0
# ---------------------------------------------------------------------------

for col, dist in enumerate(scan_distances):
    ex = next(
        (g for g in groups
         if g["scan_location"] == 1
         and g["marker_id"]    == 0
         and g["scan_distance"] == dist),
        None,
    )

    # --- Position deviation ---
    fig_ts_pos = plt.figure(figsize=FIG_SIZE)
    ax_pos = fig_ts_pos.add_subplot(111)

    if ex is None:
        ax_pos.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_pos.transAxes)
    else:
        fn = np.arange(ex["n"])
        for i, (lbl, c) in enumerate(zip(XYZ_LABELS, XYZ_COLORS)):
            ax_pos.plot(fn, ex["pos_dev_xyz"][:, i] * 1000, color=c,
                        linewidth=0.8, label=lbl)
        ax_pos.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax_pos.legend(loc="upper right", fontsize=7)

    ax_pos.set_xlabel("Frame index")
    ax_pos.set_ylabel("Position deviation [mm]")
    plt.tight_layout()
    out_ts_pos = os.path.join(script_dir,
        f"marker_noise_timeseries_pos_{int(round(dist * 100))}cm.png")
    plt.savefig(out_ts_pos, dpi=150, bbox_inches="tight")
    print(f"Saved {out_ts_pos}")

    # --- Orientation deviation ---
    fig_ts_ori = plt.figure(figsize=FIG_SIZE)
    ax_ori = fig_ts_ori.add_subplot(111)

    if ex is not None:
        fn = np.arange(ex["n"])
        ax_ori.plot(fn, ex["ori_dev"], color=DIST_COLORS[dist], linewidth=0.8)
        ax_ori.axhline(0, color="k", linewidth=0.5, linestyle="--")
    else:
        ax_ori.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_ori.transAxes)

    ax_ori.set_xlabel("Frame index")
    ax_ori.set_ylabel("Orientation deviation [°]")
    plt.tight_layout()
    out_ts_ori = os.path.join(script_dir,
        f"marker_noise_timeseries_ori_{int(round(dist * 100))}cm.png")
    plt.savefig(out_ts_ori, dpi=150, bbox_inches="tight")
    print(f"Saved {out_ts_ori}")

# ---------------------------------------------------------------------------
# Figure 2 - Power Spectral Density (Welch), averaged by scan_distance
#   Left:  position deviation magnitude  [m² / (cycles/frame)]
#   Right: orientation deviation          [deg² / (cycles/frame)]
#   Shaded band = ±1 σ across all groups at that distance
# ---------------------------------------------------------------------------

NPERSEG = 32   # segment length - fixed so all groups share the same freq grid

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

# --- PSD: Position ---
fig2_pos = plt.figure(figsize=FIG_SIZE)
ax_pp = fig2_pos.add_subplot(111)

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]
    if not psd_pos[dist]:
        continue
    arr_pos = np.array(psd_pos[dist])
    mean_ = arr_pos.mean(axis=0)
    std_  = arr_pos.std(axis=0)
    ax_pp.semilogy(freqs, mean_, color=color, linewidth=1.5, label=lbl)
    ax_pp.fill_between(
        freqs,
        np.clip(mean_ - std_, 1e-15, None),
        mean_ + std_,
        color=color, alpha=0.2,
    )

ax_pp.set_xlabel("Normalized frequency [cycles/frame]")
ax_pp.set_ylabel("PSD [m²/(cycles/frame)]")
ax_pp.grid(True, which="both", alpha=0.3)
ax_pp.legend(title="Scan distance")
plt.tight_layout()
out2_pos = os.path.join(script_dir, "marker_noise_psd_pos.png")
plt.savefig(out2_pos, dpi=150, bbox_inches="tight")
print(f"Saved {out2_pos}")

# --- PSD: Orientation ---
fig2_ori = plt.figure(figsize=FIG_SIZE)
ax_po = fig2_ori.add_subplot(111)

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]
    if not psd_ori[dist]:
        continue
    arr_ori = np.array(psd_ori[dist])
    mean_ = arr_ori.mean(axis=0)
    std_  = arr_ori.std(axis=0)
    ax_po.semilogy(freqs, mean_, color=color, linewidth=1.5, label=lbl)
    ax_po.fill_between(
        freqs,
        np.clip(mean_ - std_, 1e-15, None),
        mean_ + std_,
        color=color, alpha=0.2,
    )

ax_po.set_xlabel("Normalized frequency [cycles/frame]")
ax_po.set_ylabel("PSD [deg²/(cycles/frame)]")
ax_po.grid(True, which="both", alpha=0.3)
ax_po.legend(title="Scan distance")
plt.tight_layout()
out2_ori = os.path.join(script_dir, "marker_noise_psd_ori.png")
plt.savefig(out2_ori, dpi=150, bbox_inches="tight")
print(f"Saved {out2_ori}")

# ---------------------------------------------------------------------------
# Figure 3 - Frame-to-frame jump distributions (violin plots)
#   Left:  position jump magnitude (mm)
#   Right: orientation jump magnitude (°)
#   x-axis: scan distance   (all files and markers combined)
# ---------------------------------------------------------------------------

all_pos_deltas = {d: [] for d in scan_distances}
all_ori_deltas = {d: [] for d in scan_distances}

for g in groups:
    all_pos_deltas[g["scan_distance"]].extend(g["pos_delta"].tolist())
    all_ori_deltas[g["scan_distance"]].extend(g["ori_delta"].tolist())

x_labels = [f"{int(round(d * 100))}" for d in scan_distances]
positions = range(len(scan_distances))

# --- Violin: Position jumps ---
fig3_pos = plt.figure(figsize=FIG_SIZE)
ax_dp = fig3_pos.add_subplot(111)
scale = 1000.0
data_list = [np.array(all_pos_deltas[d]) * scale for d in scan_distances]
vp = ax_dp.violinplot(data_list, positions=positions,
                      showmedians=True, showextrema=True)
for body, dist in zip(vp["bodies"], scan_distances):
    body.set_facecolor(DIST_COLORS[dist])
    body.set_alpha(0.7)
ax_dp.set_xticks(positions)
ax_dp.set_xticklabels(x_labels)
ax_dp.set_xlabel("Scan distance [cm]")
ax_dp.set_ylabel("Frame-to-frame Δ [mm]")
ax_dp.grid(True, axis="y", alpha=0.3)
for xi, dist in zip(positions, scan_distances):
    med = np.median(all_pos_deltas[dist]) * scale
    ax_dp.text(xi, ax_dp.get_ylim()[1] * 0.97, f"med={med:.3g}",
               ha="center", va="top", fontsize=7)
plt.tight_layout()
out3_pos = os.path.join(script_dir, "marker_noise_deltas_pos.png")
plt.savefig(out3_pos, dpi=150, bbox_inches="tight")
print(f"Saved {out3_pos}")

# --- Violin: Orientation jumps ---
fig3_ori = plt.figure(figsize=FIG_SIZE)
ax_do = fig3_ori.add_subplot(111)
data_list = [np.array(all_ori_deltas[d]) for d in scan_distances]
vp = ax_do.violinplot(data_list, positions=positions,
                      showmedians=True, showextrema=True)
for body, dist in zip(vp["bodies"], scan_distances):
    body.set_facecolor(DIST_COLORS[dist])
    body.set_alpha(0.7)
ax_do.set_xticks(positions)
ax_do.set_xticklabels(x_labels)
ax_do.set_xlabel("Scan distance [cm]")
ax_do.set_ylabel("Frame-to-frame Δ [°]")
ax_do.grid(True, axis="y", alpha=0.3)
for xi, dist in zip(positions, scan_distances):
    med = np.median(all_ori_deltas[dist])
    ax_do.text(xi, ax_do.get_ylim()[1] * 0.97, f"med={med:.3g}",
               ha="center", va="top", fontsize=7)
plt.tight_layout()
out3_ori = os.path.join(script_dir, "marker_noise_deltas_ori.png")
plt.savefig(out3_ori, dpi=150, bbox_inches="tight")
print(f"Saved {out3_ori}")

# ---------------------------------------------------------------------------
# Figure 4 - Orientation noise: before vs after a Butterworth low-pass filter
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

# --- Figure 4a: PSD raw vs filtered ---
fig4l = plt.figure(figsize=FIG_SIZE)
ax4l = fig4l.add_subplot(111)

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]
    if psd_ori_raw[dist]:
        raw_mean  = np.array(psd_ori_raw [dist]).mean(axis=0)
        filt_mean = np.array(psd_ori_filt[dist]).mean(axis=0)
        ax4l.semilogy(f, raw_mean,  color=color, linewidth=1.8,
                      linestyle="-",  label=f"{lbl} raw")
        ax4l.semilogy(f, filt_mean, color=color, linewidth=1.8,
                      linestyle="--", label=f"{lbl} filtered")

ax4l.axvline(FC_DEMO, color="k", linewidth=1.0, linestyle="--",
             label=f"f_c={FC_DEMO} ({FC_DEMO * FPS / 2:.2g} Hz)")
ax4l.set_xlabel("Normalized frequency [cycles/frame]")
ax4l.set_ylabel("PSD [deg²/(cycles/frame)]")
ax4l.legend(fontsize=7, ncol=2)
ax4l.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out4l = os.path.join(script_dir, "marker_noise_lpf_psd.png")
plt.savefig(out4l, dpi=150, bbox_inches="tight")
print(f"Saved {out4l}")

# --- Figure 4b: RMS retained vs cutoff ---
fig4r = plt.figure(figsize=FIG_SIZE)
ax4r = fig4r.add_subplot(111)

for dist in scan_distances:
    lbl   = f"{int(round(dist * 100))} cm"
    color = DIST_COLORS[dist]
    if rms_filt_curves[dist]:
        raw_rms_mean  = np.mean(rms_raw[dist])
        filt_arr      = np.array(rms_filt_curves[dist])
        filt_mean_    = filt_arr.mean(axis=0)
        filt_std_     = filt_arr.std(axis=0)
        ax4r.axhline(raw_rms_mean, color=color, linewidth=1.0,
                     linestyle=":", alpha=0.7)
        ax4r.plot(CUTOFFS, filt_mean_, color=color, linewidth=1.8, label=lbl)
        ax4r.fill_between(CUTOFFS,
                          filt_mean_ - filt_std_,
                          filt_mean_ + filt_std_,
                          color=color, alpha=0.15)

ax4r.axvline(FC_DEMO, color="k", linewidth=1.0, linestyle="--",
             label=f"f_c={FC_DEMO} ({FC_DEMO * FPS / 2:.2g} Hz)")
for fc_ann, ls in ((0.01, ":"), (0.05, "-."), (0.1, ":")):
    ax4r.axvline(fc_ann, color="grey", linewidth=0.8, linestyle=ls)
    ax4r.text(fc_ann + 0.002,
              ax4r.get_ylim()[0] if ax4r.get_ylim()[0] > 0 else 0,
              f"{fc_ann}\n({fc_ann * FPS / 2:.2g} Hz)",
              color="grey", fontsize=6, va="bottom")

ax4r.set_xlabel(f"Low-pass cutoff Wn [normalized, 1=Nyquist={FPS/2:.0f} Hz]")
ax4r.set_ylabel("Orientation RMS [°]")
ax4r.legend(title="Scan dist. (dotted=unfiltered)", fontsize=7)
ax4r.grid(True, alpha=0.3)
plt.tight_layout()
out4r = os.path.join(script_dir, "marker_noise_lpf_rms.png")
plt.savefig(out4r, dpi=150, bbox_inches="tight")
print(f"Saved {out4r}")

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
