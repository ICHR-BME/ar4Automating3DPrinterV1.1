#!/usr/bin/env python3
"""
Visualise scan_raw_measurements.csv.

Two 3-D scatter plots:
  1. Position noise: each (marker_id, scan_distance, movement_id) group is
     mean-centred on Z (removes depth-shift from robot moving closer).  X and Y
     are left as absolute values so lateral systematic errors stay visible.
     Different robot movements to the same distance are shown with distinct
     scatter marker shapes so positioning error between approaches is visible.
  2. Orientation spread as rotation vectors relative to each group's mean
     orientation (computed per (marker_id, scan_distance, movement_id) so
     cross-movement positioning error doesn't inflate the spread).
     Each point is the 3-D rotation vector of q_mean^-1 * q_i:
       direction = axis of rotation from the mean
       magnitude = angle of rotation from the mean (radians)
     This avoids Euler-angle discontinuities / gimbal lock entirely.

Colour scheme
  - Each marker_id gets a distinct base hue (from a qualitative palette).
  - Each unique scan_distance for that marker gets a different shade/lightness
    of that hue (darker = closer, lighter = farther).
  - Each unique movement_id gets a distinct scatter marker shape.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from scipy.spatial.transform import Rotation as R
import colorsys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── load data ──────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_REPO_ROOT, "data", "logs", "rawMeasurementsDebug.csv")

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
required = {'marker_id', 'scan_distance', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'}
missing = required - set(df.columns)
if missing:
    print(f"Error: CSV is missing columns: {missing}")
    sys.exit(1)

# Backwards-compat: old CSVs without movement_id get a single movement group of 0
if 'movement_id' not in df.columns:
    df['movement_id'] = 0

# Convert quaternions to Euler angles (degrees)
quats = df[['qx', 'qy', 'qz', 'qw']].values

# X and Y are mean-centred on the per-marker_id mean so the scatter shows systematic
# lateral bias relative to each marker's own average, while preserving cross-movement
# and cross-distance differences.  Z is mean-centred per (marker, distance, movement)
# group to remove the depth-shift systematic.
df['_dx'] = 0.0
df['_dy'] = 0.0
df['_dz'] = 0.0
for mid, grp in df.groupby('marker_id'):
    df.loc[grp.index, '_dx'] = grp['px'] - grp['px'].mean()
    df.loc[grp.index, '_dy'] = grp['py'] - grp['py'].mean()

for (mid, dist, mvid), grp in df.groupby(['marker_id', 'scan_distance', 'movement_id']):
    df.loc[grp.index, '_dz'] = grp['pz'] - grp['pz'].mean()

# Compute per-marker mean quaternion and relative rotation vectors
# Mean quaternion: iterative weighted average on the rotation manifold using
# SLERP-based tangent-space averaging (robust to antipodal sign flips).
def _mean_quaternion(qs):
    """Return the geodesic mean of an array of unit quaternions (N x 4, [x,y,z,w])."""
    # Flip quaternions to be on the same hemisphere as the first sample
    ref = qs[0].copy()
    signs = np.sign(np.sum(qs * ref, axis=1))  # dot product with ref
    signs[signs == 0] = 1
    qs_aligned = qs * signs[:, None]
    mean_q = qs_aligned.mean(axis=0)
    norm = np.linalg.norm(mean_q)
    return mean_q / norm if norm > 0 else mean_q

df['_rv_x'] = 0.0
df['_rv_y'] = 0.0
df['_rv_z'] = 0.0

for (mid, dist, mvid), grp in df.groupby(['marker_id', 'scan_distance', 'movement_id']):
    qs = grp[['qx', 'qy', 'qz', 'qw']].values
    q_mean = _mean_quaternion(qs)
    R_mean_inv = R.from_quat(q_mean).inv()
    # Relative rotation for each sample: R_mean^-1 * R_i  →  rotation vector
    R_rel = R_mean_inv * R.from_quat(qs)
    rvecs = R_rel.as_rotvec()   # shape (N, 3), units = radians
    df.loc[grp.index, '_rv_x'] = rvecs[:, 0]
    df.loc[grp.index, '_rv_y'] = rvecs[:, 1]
    df.loc[grp.index, '_rv_z'] = rvecs[:, 2]

# Per-group scalar errors used for FFT.
# Position error: 3-D Euclidean distance from the group mean position.
# Orientation error: rotation angle (radians) from the group mean orientation.
df['_pos_err'] = 0.0
for (mid, dist, mvid), grp in df.groupby(['marker_id', 'scan_distance', 'movement_id']):
    ep = grp[['px', 'py', 'pz']].values - grp[['px', 'py', 'pz']].values.mean(axis=0)
    df.loc[grp.index, '_pos_err'] = np.linalg.norm(ep, axis=1)

df['_ori_err'] = np.sqrt(df['_rv_x']**2 + df['_rv_y']**2 + df['_rv_z']**2)

# ── colour mapping ─────────────────────────────────────────────────────────────
# One base hue per marker; shades along that hue for each distance.
MARKER_BASE_HUES = [0.60, 0.95, 0.35, 0.15, 0.75, 0.50]   # up to 6 markers; extend if needed
marker_ids     = sorted(df['marker_id'].unique())
n_markers      = len(marker_ids)

def _shades_for_marker(hue, distances_sorted):
    """Return a dict {distance: rgba} with lightness decreasing as distance decreases."""
    n = len(distances_sorted)
    if n == 1:
        lightness_values = [0.45]
    else:
        # lightness range [0.30 (darkest/closest) … 0.70 (lightest/farthest)]
        # distances_sorted is ascending (closer = darker)
        lightness_values = np.linspace(0.30, 0.70, n)
    colours = {}
    for dist, lum in zip(distances_sorted, lightness_values):
        rgb = colorsys.hls_to_rgb(hue, lum, 0.80)
        colours[dist] = rgb
    return colours

marker_colour_map = {}   # marker_id -> {distance -> rgba}
for i, mid in enumerate(marker_ids):
    hue = MARKER_BASE_HUES[i % len(MARKER_BASE_HUES)]
    dists = sorted(df[df['marker_id'] == mid]['scan_distance'].unique())
    marker_colour_map[mid] = _shades_for_marker(hue, dists)

df['_color'] = pd.Series(
    [marker_colour_map[mid][d] for mid, d in zip(df['marker_id'], df['scan_distance'])],
    index=df.index, dtype=object,
)

# Movement shapes — each unique movement_id gets a distinct matplotlib marker shape
MOVEMENT_MARKERS = ['o', '^', 's', 'D', 'v', 'P', '*', 'X']
movement_ids = sorted(df['movement_id'].unique())

# ── legend handles ─────────────────────────────────────────────────────────────
def build_legend_handles():
    handles = []
    # Colour entries: marker × distance
    for mid in marker_ids:
        dist_map = marker_colour_map[mid]
        for dist in sorted(dist_map):
            colour = dist_map[dist]
            label  = f"ID {mid}  d={dist:.3f} m"
            handles.append(
                mlines.Line2D([], [], color=colour, marker='o', linestyle='None',
                              markersize=7, label=label)
            )
    # Shape entries: movement_id
    if len(movement_ids) > 1:
        handles.append(mpatches.Patch(color='none', label=''))  # spacer
        for mvid in movement_ids:
            shape = MOVEMENT_MARKERS[mvid % len(MOVEMENT_MARKERS)]
            handles.append(
                mlines.Line2D([], [], color='gray', marker=shape, linestyle='None',
                              markersize=7, label=f'Movement {mvid}')
            )
    return handles

legend_handles = build_legend_handles()

# ── helper: draw one scatter ───────────────────────────────────────────────────
def scatter3(ax, df_plot, col_x, col_y, col_z, xlabel, ylabel, zlabel, title):
    """Scatter by movement_id so each movement gets a distinct marker shape."""
    for mvid in movement_ids:
        sub = df_plot[df_plot['movement_id'] == mvid]
        if sub.empty:
            continue
        shape = MOVEMENT_MARKERS[mvid % len(MOVEMENT_MARKERS)]
        ax.scatter(sub[col_x], sub[col_y], sub[col_z],
                   c=list(sub['_color']), s=18, depthshade=True,
                   edgecolors='none', alpha=0.85, marker=shape)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_zlabel(zlabel, labelpad=6)
    ax.set_title(title)

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.suptitle("ArUco raw scan measurements", fontsize=13, fontweight='bold')

# --- Plot 1: positions (X/Y raw, Z mean-centred per group) ---
ax1 = fig.add_subplot(121, projection='3d')
scatter3(ax1, df, '_dx', '_dy', '_dz',
         'ΔX (m)', 'ΔY (m)', 'ΔZ (m)',
         'Position error (X/Y global-mean-centred, Z per-movement-mean-centred)')

# --- Plot 2: orientation spread as rotation vectors from per-movement mean ---
ax2 = fig.add_subplot(122, projection='3d')
scatter3(ax2, df, '_rv_x', '_rv_y', '_rv_z',
         'RV x (rad)', 'RV y (rad)', 'RV z (rad)',
         'Orientation spread\n(rotation vector from per-movement mean)')

# shared legend to the right of the figure
fig.legend(handles=legend_handles, loc='center right',
           title='Marker / distance', fontsize=8, title_fontsize=9,
           bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 0.82, 1.0])

# ── FFT of measurement error ───────────────────────────────────────────────────
# For each (marker_id, scan_distance, movement_id) group, treat the sequence of
# per-frame scalar errors as a time-series (frames assumed uniformly sampled) and
# plot the one-sided amplitude spectrum.  DC (index 0) is omitted because it just
# reflects the mean error level, not the frequency content of the noise.
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("FFT of measurement error (per movement group)",
              fontsize=13, fontweight='bold')

for (mid, dist, mvid), grp in df.groupby(['marker_id', 'scan_distance', 'movement_id']):
    n = len(grp)
    if n < 4:
        continue  # too few points for a meaningful spectrum
    colour = marker_colour_map[mid][dist]
    linestyle = ['-', '--', ':', '-.'][mvid % 4]
    label = f"ID {mid}  d={dist:.3f} m  mv={mvid}"
    freqs = np.fft.rfftfreq(n)  # normalised: 0 … 0.5 cycles/frame

    pos_amp = np.abs(np.fft.rfft(grp['_pos_err'].values))
    ax3.plot(freqs[1:], pos_amp[1:], color=colour, linestyle=linestyle,
             linewidth=1.2, alpha=0.85, label=label)

    ori_amp = np.abs(np.fft.rfft(grp['_ori_err'].values))
    ax4.plot(freqs[1:], ori_amp[1:], color=colour, linestyle=linestyle,
             linewidth=1.2, alpha=0.85, label=label)

ax3.set_xlabel('Frequency (cycles / frame)')
ax3.set_ylabel('Amplitude (m)')
ax3.set_title('Position error spectrum')
ax3.grid(True, alpha=0.3)

ax4.set_xlabel('Frequency (cycles / frame)')
ax4.set_ylabel('Amplitude (rad)')
ax4.set_title('Orientation error spectrum')
ax4.grid(True, alpha=0.3)

# Build a compact legend: one entry per (marker, distance) with shape entries for movements
fft_handles = []
for mid in marker_ids:
    dist_map = marker_colour_map[mid]
    for dist in sorted(dist_map):
        colour = dist_map[dist]
        fft_handles.append(
            mlines.Line2D([], [], color=colour, linestyle='-', linewidth=2,
                          label=f"ID {mid}  d={dist:.3f} m")
        )
if len(movement_ids) > 1:
    fft_handles.append(mpatches.Patch(color='none', label=''))  # spacer
    for mvid in movement_ids:
        ls = ['-', '--', ':', '-.'][mvid % 4]
        fft_handles.append(
            mlines.Line2D([], [], color='gray', linestyle=ls, linewidth=2,
                          label=f'Movement {mvid}')
        )
fig2.legend(handles=fft_handles, loc='center right',
            title='Marker / distance', fontsize=8, title_fontsize=9,
            bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 0.82, 1.0])

# ── summary stats ──────────────────────────────────────────────────────────────
print(f"\nLoaded {len(df)} measurements")
print(f"Markers: {marker_ids}")
print(f"Movements: {movement_ids}")
for mid in marker_ids:
    sub = df[df['marker_id'] == mid]
    print(f"\n  Marker {mid}  ({len(sub)} frames)")
    for dist in sorted(sub['scan_distance'].unique()):
        d = sub[sub['scan_distance'] == dist]
        for mvid in sorted(d['movement_id'].unique()):
            m = d[d['movement_id'] == mvid]
            rv_mag = np.sqrt(m['_rv_x']**2 + m['_rv_y']**2 + m['_rv_z']**2)
            pos_noise = np.sqrt(m['_dx']**2 + m['_dy']**2 + m['_dz']**2)
            print(f"    dist={dist:.4f} m  movement={mvid}  n={len(m)}"
                  f"  pos_noise_std=[{m['_dx'].std():.4f}, {m['_dy'].std():.4f}, {m['_dz'].std():.4f}]"
                  f"  pos_rms={pos_noise.mean():.4f} m"
                  f"  orient_angle_std={np.degrees(rv_mag.std()):.2f}°")

plt.show()
