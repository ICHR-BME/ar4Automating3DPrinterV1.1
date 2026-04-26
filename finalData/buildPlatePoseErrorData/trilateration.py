import numpy as np
import csv
import os

# ── Configuration ──────────────────────────────────────────────────────────────
RUN_TEST   = False          # True  → synthetic self-test
                            # False → load measurements from CSV_FILE
CSV_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buildPlatePlacementErrorMeasurements3.csv")
CORNER_SPACING = 198.0      # L: distance between the two reference corners (meters)
# ──────────────────────────────────────────────────────────────────────────────

def solve_rigid_body(d1, d2, d_cross1, d_cross2, L):
    """
    Recovers orientation offset and centroid translation magnitude from
    4 ruler measurements and known corner spacing L. No pivot point needed.

    d1:       distance from c1 original mark -> c1 new position
    d2:       distance from c2 original mark -> c2 new position
    d_cross1: distance from c1 original mark -> c2 new position
    d_cross2: distance from c2 original mark -> c1 new position
    L:        corner spacing |c2 - c1| (rigid body constant)

    Derivation (frame with c1=origin, c2=(L,0)):
      cos(theta) = (d_cross1^2 + d_cross2^2 - d1^2 - d2^2) / (2*L^2)
      |centroid_offset|^2 = (d1^2 + d2^2)/2 - L^2*(1 - cos(theta))/2
    """
    cos_theta = (d_cross1**2 + d_cross2**2 - d1**2 - d2**2) / (2 * L**2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # guard against floating-point overshoot
    theta_deg = np.degrees(np.arccos(cos_theta))

    centroid_offset = np.sqrt(max(0.0, (d1**2 + d2**2) / 2 - L**2 * (1 - cos_theta) / 2))

    return theta_deg, centroid_offset


def run_test():
    """Synthetic self-test: generate measurements from known ground truth and verify recovery."""
    true_theta_deg = 0.0
    true_tx, true_ty = 1.0, 0.0
    c1 = np.array([5.0,  5.0])
    c2 = np.array([5.0, -5.0])

    L = np.linalg.norm(c2 - c1)
    rad = np.radians(true_theta_deg)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad),  np.cos(rad)]])
    T = np.array([true_tx, true_ty])

    c1_new = R @ c1 + T
    c2_new = R @ c2 + T

    d1       = np.linalg.norm(c1_new - c1)
    d2       = np.linalg.norm(c2_new - c2)
    d_cross1 = np.linalg.norm(c2_new - c1)
    d_cross2 = np.linalg.norm(c1_new - c2)

    centroid_orig = (c1 + c2) / 2
    centroid_new  = (c1_new + c2_new) / 2
    true_centroid_offset = np.linalg.norm(centroid_new - centroid_orig)

    print("--- FORWARD STEP (Measurements) ---")
    print(f"True Rotation:         {true_theta_deg}°")
    print(f"True Translation:      ({true_tx}, {true_ty})")
    print(f"True Centroid Offset:  {true_centroid_offset:.6f}")
    print(f"d1       = {d1:.6f}")
    print(f"d2       = {d2:.6f}")
    print(f"d_cross1 = {d_cross1:.6f}")
    print(f"d_cross2 = {d_cross2:.6f}\n")

    theta_recovered, centroid_recovered = solve_rigid_body(d1, d2, d_cross1, d_cross2, L)

    print("--- INVERSE STEP (Recovery) ---")
    print(f"Recovered Rotation:        {theta_recovered:.6f}°  (true: {true_theta_deg}°)")
    print(f"Recovered Centroid Offset: {centroid_recovered:.6f}  (true: {true_centroid_offset:.6f})")
    print(f"Rotation error:            {abs(theta_recovered - true_theta_deg):.2e}°")
    print(f"Centroid offset error:     {abs(centroid_recovered - true_centroid_offset):.2e}")


def run_csv():
    """
    Load measurements from CSV_FILE. Each row has 4 columns:
      d1, d_cross2, d2, d_cross1
      (point1_displacement, point1_diagonal, point2_displacement, point2_diagonal)
    Appends two new columns (orientation_deg, centroid_offset) and prints a table.
    """
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found: {CSV_FILE}")
        return

    rows = []
    with open(CSV_FILE, newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or line[0].strip().startswith("#"):
                continue
            rows.append([col.strip() for col in line])

    if not rows:
        print("ERROR: CSV file is empty or has no data rows.")
        return

    # Detect header row
    header = None
    data_rows = rows
    try:
        float(rows[0][0])
    except ValueError:
        header = rows[0]
        data_rows = rows[1:]

    results = []
    for i, row in enumerate(data_rows):
        if len(row) < 4:
            print(f"WARNING: row {i+1} has fewer than 4 columns, skipping.")
            continue
        try:
            d1, d_cross2, d2, d_cross1 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        except ValueError as e:
            print(f"WARNING: row {i+1} could not be parsed ({e}), skipping.")
            continue
        theta_deg, centroid_offset = solve_rigid_body(d1, d2, d_cross1, d_cross2, CORNER_SPACING)
        results.append((row, theta_deg, centroid_offset))

    # Print table
    col_w = 16
    h1 = "d1"
    h2 = "d_cross2"
    h3 = "d2"
    h4 = "d_cross1"
    h5 = "orientation (°)"
    h6 = "centroid offset"
    print(f"\n{'Row':<5} {h1:>{col_w}} {h2:>{col_w}} {h3:>{col_w}} {h4:>{col_w}} {h5:>{col_w}} {h6:>{col_w}}")
    print("-" * (5 + (col_w + 1) * 6))
    for i, (row, theta, offset) in enumerate(results):
        print(f"{i+1:<5} {row[0]:>{col_w}} {row[1]:>{col_w}} {row[2]:>{col_w}} {row[3]:>{col_w}} "
              f"{theta:>{col_w}.4f} {offset:>{col_w}.4f}")

    # Write results to a new CSV file (original data is preserved)
    base, ext = os.path.splitext(CSV_FILE)
    out_file = base + "_results" + ext

    out_rows = []
    if header is not None:
        while len(header) < 4:
            header.append("")
        out_rows.append(header + ["orientation_deg", "centroid_offset"])
    else:
        out_rows.append(["d1", "d_cross2", "d2", "d_cross1", "orientation_deg", "centroid_offset"])
    for row, theta, offset in results:
        out_rows.append(list(row) + [f"{theta:.6f}", f"{offset:.6f}"])

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    print(f"\nResults written to: {out_file}")


if __name__ == "__main__":
    if RUN_TEST:
        run_test()
    else:
        run_csv()
