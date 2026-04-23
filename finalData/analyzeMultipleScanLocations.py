import os
import glob
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation

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

BASE_COLORS = {mid: c for mid, c in zip(marker_ids, ["tab:red", "tab:blue", "tab:green"])}

# Lighter shades for shorter (closer) distances, darker for longer distances
lightness_levels = np.linspace(0.75, 0.25, len(distances))   # 0.75 → 0.25
dist_lightness   = {d: lightness_levels[i] for i, d in enumerate(distances)}

def point_color(marker_id: int, scan_distance: float):
    return adjust_lightness(BASE_COLORS[marker_id], dist_lightness[scan_distance])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(17, 7))
ax_pos = fig.add_subplot(121, projection="3d")
ax_ori = fig.add_subplot(122, projection="3d")

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
        ax_ori.scatter(
            subset["roll_deg"], subset["pitch_deg"], subset["yaw_deg"],
            color=color, s=8, depthshade=False, linewidths=0,
        )

# Axis labels
ax_pos.set_xlabel("X (m)")
ax_pos.set_ylabel("Y (m)")
ax_pos.set_zlabel("Z (m)")
ax_pos.set_title("Marker Positions")

ax_ori.set_xlabel("Roll (°)")
ax_ori.set_ylabel("Pitch (°)")
ax_ori.set_zlabel("Yaw (°)")
ax_ori.set_title("Marker Orientations (Euler XYZ)")

# ---------------------------------------------------------------------------
# Legend  –  two sections: marker ID (color) and distance (shade)
# ---------------------------------------------------------------------------

color_handles = [
    mpatches.Patch(facecolor=BASE_COLORS[mid], label=f"Marker {mid}")
    for mid in marker_ids
]

separator = mpatches.Patch(visible=False, label="")

shade_handles = [
    mpatches.Patch(facecolor=(lv, lv, lv), label=f"{int(round(d * 100))} cm")
    for d, lv in zip(distances, lightness_levels)
]

legend = fig.legend(
    handles=color_handles + [separator] + shade_handles,
    loc="lower center",
    ncol=len(color_handles) + 1 + len(shade_handles),
    bbox_to_anchor=(0.5, -0.01),
    frameon=True,
    title="Color = Marker ID                                    Shade = Scan Distance",
    title_fontsize=9,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

out_path = os.path.join(script_dir, "marker_pose_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
plt.show()

# ---------------------------------------------------------------------------
# Compute per-(file, marker_id, scan_distance) means
#   Positions: mean of px, py, pz
#   Orientations: average quaternion (renormalised), then to Euler
# ---------------------------------------------------------------------------

scan_locations = sorted(data["scan_location"].unique())

# Marker shape per file so individual files are distinguishable
FILE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

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

# ---------------------------------------------------------------------------
# Mean plots
# ---------------------------------------------------------------------------

fig2 = plt.figure(figsize=(17, 7))
ax_mpos = fig2.add_subplot(121, projection="3d")
ax_mori = fig2.add_subplot(122, projection="3d")

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
            mk = FILE_MARKERS[(loc - 1) % len(FILE_MARKERS)]
            ax_mpos.scatter(
                row["px"], row["py"], row["pz"],
                color=color, s=60, marker=mk, depthshade=False, linewidths=0.5,
                edgecolors="k",
            )
            ax_mori.scatter(
                row["roll_deg"], row["pitch_deg"], row["yaw_deg"],
                color=color, s=60, marker=mk, depthshade=False, linewidths=0.5,
                edgecolors="k",
            )

ax_mpos.set_xlabel("X (m)")
ax_mpos.set_ylabel("Y (m)")
ax_mpos.set_zlabel("Z (m)")
ax_mpos.set_title("Mean Marker Positions (per file)")

ax_mori.set_xlabel("Roll (°)")
ax_mori.set_ylabel("Pitch (°)")
ax_mori.set_zlabel("Yaw (°)")
ax_mori.set_title("Mean Marker Orientations (per file, Euler XYZ)")

# Legend: colors (marker IDs), shades (distances), markers (files)
file_handles = [
    plt.Line2D([0], [0],
               marker=FILE_MARKERS[(loc - 1) % len(FILE_MARKERS)],
               color="w", markerfacecolor="grey", markeredgecolor="k",
               markersize=8, label=f"File {loc}")
    for loc in scan_locations
]

legend2 = fig2.legend(
    handles=color_handles + [separator] + shade_handles + [separator] + file_handles,
    loc="lower center",
    ncol=len(color_handles) + 1 + len(shade_handles) + 1 + len(file_handles),
    bbox_to_anchor=(0.5, -0.01),
    frameon=True,
    title=(
        "Color = Marker ID          "
        "Shade = Scan Distance          "
        "Shape = File"
    ),
    title_fontsize=9,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

out_path2 = os.path.join(script_dir, "marker_pose_means.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path2}")
plt.show()

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

# Spread file markers along the x-axis so error bars don't collide
n_files     = len(scan_locations)
x_jitter    = np.linspace(-0.003, 0.003, n_files)   # metres
gray_levels = np.linspace(0.15, 0.65, n_files)       # darkest → lightest

fig3, (ax_pe, ax_oe) = plt.subplots(1, 2, figsize=(14, 6))

for i, loc in enumerate(scan_locations):
    loc_err = errors[errors["scan_location"] == loc]
    x = loc_err["scan_distance"].values + x_jitter[i]
    g = gray_levels[i]
    mk = FILE_MARKERS[(loc - 1) % len(FILE_MARKERS)]

    ax_pe.errorbar(
        x, loc_err["pos_error_mean"], yerr=loc_err["pos_error_std"],
        fmt=mk, color=(g, g, g), capsize=4, markersize=7,
        markeredgecolor="k", markeredgewidth=0.5,
        label=f"File {loc}",
    )
    ax_oe.errorbar(
        x, loc_err["ori_error_mean"], yerr=loc_err["ori_error_std"],
        fmt=mk, color=(g, g, g), capsize=4, markersize=7,
        markeredgecolor="k", markeredgewidth=0.5,
        label=f"File {loc}",
    )

for ax in (ax_pe, ax_oe):
    ax.set_xticks(distances)
    ax.set_xticklabels([f"{int(round(d * 100))} cm" for d in distances])
    ax.set_xlabel("Scan distance")
    ax.grid(True, alpha=0.3)
    ax.legend(title="File")

ax_pe.set_ylabel("Position error (m)")
ax_pe.set_title(
    "Position Error vs Scan Distance\n"
    "Reference = each file's own mean at closest distance (15 cm)"
)

ax_oe.set_ylabel("Orientation error (°)")
ax_oe.set_title(
    "Orientation Error vs Scan Distance\n"
    "Reference = each file's own mean at closest distance (15 cm)"
)

plt.tight_layout()

out_path3 = os.path.join(script_dir, "marker_pose_errors.png")
plt.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path3}")
plt.show()
