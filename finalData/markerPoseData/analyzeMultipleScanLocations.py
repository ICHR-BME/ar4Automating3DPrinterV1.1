import os
import glob
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.labelpad": 2,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
})
FIG_SIZE      = (3.5, 3.0)
FIG_SIZE_ERR  = (3.5, 4.0)   # single panel with legend below

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adjust_lightness(color: str, lightness: float):
    """Return an RGB tuple with the hue/saturation of *color* at *lightness* [0, 1]."""
    r, g, b = plt.matplotlib.colors.to_rgb(color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return r2, g2, b2


def quat_mean(qs: np.ndarray) -> np.ndarray:
    """Hemisphere-consistent quaternion mean.

    Flips any quaternion whose dot product with the first is negative before
    averaging, then renormalises.  This prevents cancellation near the ±180°
    boundary where q and -q represent the same rotation.
    """
    qs = qs.copy()
    dots = qs @ qs[0]
    qs[dots < 0] *= -1
    q_mean = qs.mean(axis=0)
    return q_mean / np.linalg.norm(q_mean)


def quat_angular_error_deg(qs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """SLERP geodesic angle (degrees) between each row of *qs* and *q_ref*.

    Uses 2·arccos(|q · q_ref|) which handles the double-cover (q ≡ -q)
    without needing explicit hemisphere alignment, and avoids the ±180°
    discontinuity present in Euler-angle difference approaches.
    """
    dots = np.clip(np.abs(qs @ q_ref), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "markerPoseData")
csv_files = sorted(glob.glob(os.path.join(data_dir, "scan_raw_measurements*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No scan_raw_measurements*.csv files found in {data_dir}")

frames = []
for idx, path in enumerate(csv_files, start=1):
    df = pd.read_csv(path)
    df["scan_location"] = idx
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Convert quaternions to Euler angles (XYZ, degrees)
# ---------------------------------------------------------------------------

quats = data[["qx", "qy", "qz", "qw"]].values
euler = Rotation.from_quat(quats).as_euler("xyz", degrees=True)
data["roll_deg"]  = euler[:, 0]
data["pitch_deg"] = euler[:, 1]
data["yaw_deg"]   = euler[:, 2]

# ---------------------------------------------------------------------------
# Color / shade scheme
#   • Color  → marker ID   (distinct hues, insensitive to shade)
#   • Shade  → scan distance (lighter = closer, darker = farther)
# ---------------------------------------------------------------------------

marker_ids = sorted(data["marker_id"].unique())
distances  = sorted(data["scan_distance"].unique())   # ascending

BASE_COLORS = {mid: c for mid, c in zip(marker_ids, ["#2166ac", "#6baed6", "#08519c"])}

# Lighter shades for shorter (closer) distances, darker for longer distances
lightness_levels = np.linspace(0.75, 0.25, len(distances))   # 0.75 → 0.25
dist_lightness   = {d: lightness_levels[i] for i, d in enumerate(distances)}

def point_color(marker_id: int, scan_distance: float):
    return adjust_lightness(BASE_COLORS[marker_id], dist_lightness[scan_distance])

# ---------------------------------------------------------------------------
# Plot 1a — Marker Positions (scatter)
# ---------------------------------------------------------------------------

fig_pos = plt.figure(figsize=FIG_SIZE)
ax_pos = fig_pos.add_subplot(111, projection="3d")

for mid in marker_ids:
    for dist in distances:
        mask   = (data["marker_id"] == mid) & (data["scan_distance"] == dist)
        subset = data[mask]
        if subset.empty:
            continue
        color = point_color(mid, dist)
        ax_pos.scatter(
            subset["px"], subset["py"], subset["pz"],
            color=color, s=8, depthshade=False, linewidths=0,
        )

ax_pos.set_xlabel("X [m]")
ax_pos.set_ylabel("Y [m]")
ax_pos.set_zlabel("Z [m]")

color_handles = [
    mpatches.Patch(facecolor=BASE_COLORS[mid], label=f"Marker {mid}")
    for mid in marker_ids
]
separator = mpatches.Patch(visible=False, label="")
shade_handles = [
    mpatches.Patch(facecolor=(lv, lv, lv), label=f"{int(round(d * 100))} cm")
    for d, lv in zip(distances, lightness_levels)
]

ax_pos.legend(
    handles=color_handles + [separator] + shade_handles,
    fontsize=8, loc="upper left",
    title="Color=Marker  Shade=Dist", title_fontsize=7,
)

plt.tight_layout(pad=0.5)
out_path_pos = os.path.join(script_dir, "marker_positions.png")
plt.savefig(out_path_pos, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path_pos}")

# ---------------------------------------------------------------------------
# Plot 1b — Marker Orientations (scatter)
# ---------------------------------------------------------------------------

fig_ori = plt.figure(figsize=FIG_SIZE)
ax_ori = fig_ori.add_subplot(111, projection="3d")

for mid in marker_ids:
    for dist in distances:
        mask   = (data["marker_id"] == mid) & (data["scan_distance"] == dist)
        subset = data[mask]
        if subset.empty:
            continue
        color = point_color(mid, dist)
        ax_ori.scatter(
            subset["roll_deg"], subset["pitch_deg"], subset["yaw_deg"],
            color=color, s=8, depthshade=False, linewidths=0,
        )

ax_ori.set_xlabel("Roll [°]")
ax_ori.set_ylabel("Pitch [°]")
ax_ori.set_zlabel("Yaw [°]")

ax_ori.legend(
    handles=color_handles + [separator] + shade_handles,
    fontsize=8, loc="upper left",
    title="Color=Marker  Shade=Dist", title_fontsize=7,
)

plt.tight_layout(pad=0.5)
out_path_ori = os.path.join(script_dir, "marker_orientations.png")
plt.savefig(out_path_ori, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path_ori}")

# ---------------------------------------------------------------------------
# Compute per-(file, marker_id, scan_distance) means
#   Positions: mean of px, py, pz
#   Orientations: average quaternion (renormalised), then to Euler
# ---------------------------------------------------------------------------

scan_locations = sorted(data["scan_location"].unique())

# Marker shape per trial so individual trials are distinguishable
TRIAL_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

mean_rows = []
for loc in scan_locations:
    for mid in marker_ids:
        for dist in distances:
            mask   = (data["scan_location"] == loc) & \
                     (data["marker_id"] == mid) & \
                     (data["scan_distance"] == dist)
            subset = data[mask]
            if subset.empty:
                continue

            mpx = subset["px"].mean()
            mpy = subset["py"].mean()
            mpz = subset["pz"].mean()

            # Average quaternion: hemisphere-consistent mean to avoid ±180° cancellation
            q_mean = quat_mean(subset[["qx", "qy", "qz", "qw"]].values)
            roll, pitch, yaw = Rotation.from_quat(q_mean).as_euler("xyz", degrees=True)

            mean_rows.append({
                "scan_location": loc,
                "marker_id":     mid,
                "scan_distance": dist,
                "px": mpx, "py": mpy, "pz": mpz,
                "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw,
            })

means = pd.DataFrame(mean_rows)

trial_handles = [
    plt.Line2D([0], [0],
               marker=TRIAL_MARKERS[(loc - 1) % len(TRIAL_MARKERS)],
               color="w", markerfacecolor="grey", markeredgecolor="k",
               markersize=8, label=f"Trial {loc}")
    for loc in scan_locations
]

# ---------------------------------------------------------------------------
# Plot 2a — Mean Marker Positions (per trial)
# ---------------------------------------------------------------------------

fig_mpos = plt.figure(figsize=FIG_SIZE)
ax_mpos = fig_mpos.add_subplot(111, projection="3d")

for mid in marker_ids:
    for dist in distances:
        color = point_color(mid, dist)
        for loc in scan_locations:
            row = means[
                (means["scan_location"] == loc) &
                (means["marker_id"] == mid) &
                (means["scan_distance"] == dist)
            ]
            if row.empty:
                continue
            mk = TRIAL_MARKERS[(loc - 1) % len(TRIAL_MARKERS)]
            ax_mpos.scatter(
                row["px"], row["py"], row["pz"],
                color=color, s=60, marker=mk, depthshade=False, linewidths=0.5,
                edgecolors="k",
            )

ax_mpos.set_xlabel("X [m]")
ax_mpos.set_ylabel("Y [m]")
ax_mpos.set_zlabel("Z [m]")

ax_mpos.legend(
    handles=color_handles + [separator] + shade_handles + [separator] + trial_handles,
    fontsize=7, loc="upper left",
    title="Color=Marker  Shade=Dist  Shape=Trial", title_fontsize=6,
)

plt.tight_layout(pad=0.5)
out_path_mpos = os.path.join(script_dir, "marker_mean_positions.png")
plt.savefig(out_path_mpos, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path_mpos}")

# ---------------------------------------------------------------------------
# Plot 2b — Mean Marker Orientations (per trial)
# ---------------------------------------------------------------------------

fig_mori = plt.figure(figsize=FIG_SIZE)
ax_mori = fig_mori.add_subplot(111, projection="3d")

for mid in marker_ids:
    for dist in distances:
        color = point_color(mid, dist)
        for loc in scan_locations:
            row = means[
                (means["scan_location"] == loc) &
                (means["marker_id"] == mid) &
                (means["scan_distance"] == dist)
            ]
            if row.empty:
                continue
            mk = TRIAL_MARKERS[(loc - 1) % len(TRIAL_MARKERS)]
            ax_mori.scatter(
                row["roll_deg"], row["pitch_deg"], row["yaw_deg"],
                color=color, s=60, marker=mk, depthshade=False, linewidths=0.5,
                edgecolors="k",
            )

ax_mori.set_xlabel("Roll [°]")
ax_mori.set_ylabel("Pitch [°]")
ax_mori.set_zlabel("Yaw [°]")

ax_mori.legend(
    handles=color_handles + [separator] + shade_handles + [separator] + trial_handles,
    fontsize=7, loc="upper left",
    title="Color=Marker  Shade=Dist  Shape=Trial", title_fontsize=6,
)

plt.tight_layout(pad=0.5)
out_path_mori = os.path.join(script_dir, "marker_mean_orientations.png")
plt.savefig(out_path_mori, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path_mori}")

# ---------------------------------------------------------------------------
# Error plots: deviation of each per-(file, distance) mean from the "true"
#   pose, defined as the grand mean across ALL files at the closest distance.
#
# Y-axis  = scalar error (position: metres, orientation: degrees)
# X-axis  = scan distance
# Each point = one (file, scan_distance) pair, averaged over marker IDs
# Error bars = std of individual per-measurement errors (spread of the raw
#              detection noise within that file/distance/marker group)
# ---------------------------------------------------------------------------

closest_dist = distances[0]   # 0.15 m

# True pose per (file, marker) = mean of that file's own measurements at the
# closest distance.
true_pos  = {}   # (loc, marker_id) -> ndarray(3,)
true_quat = {}   # (loc, marker_id) -> ndarray(4,)  (xyzw, unit)

for loc in scan_locations:
    for mid in marker_ids:
        sub = data[
            (data["scan_location"] == loc) &
            (data["marker_id"] == mid) &
            (data["scan_distance"] == closest_dist)
        ]
        if sub.empty:
            continue
        true_pos[(loc, mid)] = sub[["px", "py", "pz"]].values.mean(axis=0)
        true_quat[(loc, mid)] = quat_mean(sub[["qx", "qy", "qz", "qw"]].values)

# For each (scan_location, scan_distance): gather per-measurement errors
# across all marker IDs, then report mean ± std.
error_rows = []
for loc in scan_locations:
    for dist in distances:
        pos_errs = []
        ori_errs = []
        for mid in marker_ids:
            if (loc, mid) not in true_pos:
                continue
            sub = data[
                (data["scan_location"] == loc) &
                (data["marker_id"] == mid) &
                (data["scan_distance"] == dist)
            ]
            if sub.empty:
                continue

            pos_vals = sub[["px", "py", "pz"]].values
            quats    = sub[["qx", "qy", "qz", "qw"]].values

            # Per-measurement position errors
            pe = np.linalg.norm(pos_vals - true_pos[(loc, mid)], axis=1)
            pos_errs.extend(pe.tolist())

            # Per-measurement orientation errors via SLERP geodesic distance:
            # 2·arccos(|q_meas · q_ref|) handles the ±180° double-cover.
            oe = quat_angular_error_deg(quats, true_quat[(loc, mid)])
            ori_errs.extend(oe.tolist())

        if not pos_errs:
            continue

        error_rows.append({
            "scan_location":   loc,
            "scan_distance":   dist,
            "pos_error_mean":  np.mean(pos_errs),
            "pos_error_std":   np.std(pos_errs),
            "ori_error_mean":  np.mean(ori_errs),
            "ori_error_std":   np.std(ori_errs),
        })

errors = pd.DataFrame(error_rows)

n_trials    = len(scan_locations)
x_jitter    = np.linspace(-0.010, 0.010, n_trials)
blue_levels = np.linspace(0.15, 0.65, n_trials)

# ---------------------------------------------------------------------------
# Save per-distance aggregate statistics (all trials combined)
# ---------------------------------------------------------------------------

agg_rows = []
for dist in distances:
    dist_err = errors[errors["scan_distance"] == dist]
    agg_rows.append({
        "scan_distance_cm":       int(round(dist * 100)),
        "pos_error_mean_mm":      dist_err["pos_error_mean"].mean() * 1000,
        "pos_error_std_mm":       dist_err["pos_error_std"].mean() * 1000,
        "pos_error_std_trials_mm": dist_err["pos_error_mean"].std() * 1000,
        "ori_error_mean_deg":     dist_err["ori_error_mean"].mean(),
        "ori_error_std_deg":      dist_err["ori_error_std"].mean(),
        "ori_error_std_trials_deg": dist_err["ori_error_mean"].std(),
    })

agg_df = pd.DataFrame(agg_rows)
out_csv = os.path.join(script_dir, "marker_error_summary.csv")
agg_df.to_csv(out_csv, index=False, float_format="%.6f")
print(f"Saved summary CSV to {out_csv}")

# ---------------------------------------------------------------------------
# Plot 3 — Position & Orientation Error vs Scan Distance (dual y-axis)
# ---------------------------------------------------------------------------

AX_COLOR_POS = "#1a6faf"   # blue  — left axis / position error
AX_COLOR_ORI = "#cc4c02"   # orange — right axis / orientation error

fig_err, ax_left = plt.subplots(figsize=FIG_SIZE_ERR)
ax_right = ax_left.twinx()

for i, loc in enumerate(scan_locations):
    loc_err = errors[errors["scan_location"] == loc]
    x = loc_err["scan_distance"].values + x_jitter[i]
    mk = TRIAL_MARKERS[(loc - 1) % len(TRIAL_MARKERS)]

    ax_left.errorbar(
        x, loc_err["pos_error_mean"] * 1000, yerr=loc_err["pos_error_std"] * 1000,
        fmt=mk, color=AX_COLOR_POS, capsize=4, markersize=7,
        markeredgecolor="k", markeredgewidth=0.5,
        label=f"Trial {loc}",
    )
    ax_right.errorbar(
        x, loc_err["ori_error_mean"], yerr=loc_err["ori_error_std"],
        fmt=mk, color=AX_COLOR_ORI, capsize=4, markersize=7,
        markeredgecolor="k", markeredgewidth=0.5,
    )

# Stretch orientation axis so its range is 2× that of position
# (keeps the two datasets visually separated)
pos_all = errors["pos_error_mean"] * 1000
ori_all = errors["ori_error_mean"]
pos_span = pos_all.max() - pos_all.min()
ori_span = ori_all.max() - ori_all.min()
ori_pad  = max(ori_span, 0.01) * 0.5   # 50 % padding each side
ori_mid  = (ori_all.max() + ori_all.min()) / 2
ax_right.set_ylim(ori_mid - ori_span * 1, ori_mid + ori_span * 5)

ax_left.set_xticks(distances)
ax_left.set_xticklabels([f"{int(round(d * 100))}" for d in distances])
ax_left.set_xlabel("Scan distance [cm]")
ax_left.set_ylabel("Position error [mm]", color=AX_COLOR_POS)
ax_left.tick_params(axis="y", colors=AX_COLOR_POS)
ax_left.spines["left"].set_color(AX_COLOR_POS)
ax_left.grid(True, alpha=0.3)

ax_right.set_ylabel("Orientation error [°]", color=AX_COLOR_ORI)
ax_right.tick_params(axis="y", colors=AX_COLOR_ORI)
ax_right.spines["right"].set_color(AX_COLOR_ORI)

ax_left.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2, frameon=True,
)

plt.tight_layout(pad=0.5)
plt.subplots_adjust(bottom=0.30)
out_path_err = os.path.join(script_dir, "marker_error.png")
plt.savefig(out_path_err, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path_err}")

plt.show()
