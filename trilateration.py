import numpy as np

def calculate_plate_offset(theta_deg, d1, d2, corner1_ref, corner2_ref):
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    p1_rot = R @ np.array(corner1_ref)
    p2_rot = R @ np.array(corner2_ref)
    
    # Anchors: where the corner starts minus where the spin puts it
    A1 = np.array(corner1_ref) - p1_rot
    A2 = np.array(corner2_ref) - p2_rot
    
    x1, y1 = A1
    x2, y2 = A2
    dx, dy = x2 - x1, y2 - y1
    D = np.sqrt(dx**2 + dy**2)
    
    a = (d1**2 - d2**2 + D**2) / (2 * D)
    h = np.sqrt(max(0, d1**2 - a**2))
    
    x0, y0 = x1 + a * dx / D, y1 + a * dy / D
    rx, ry = -dy * (h / D), dx * (h / D)
    
    # Two possible intersection solutions
    sol1 = np.array([x0 + rx, y0 + ry])
    sol2 = np.array([x0 - rx, y0 - ry])
    
    return sol1, sol2

# --- FORWARD TRANSFORM (Generating Test Data) ---
true_theta_deg = 15.0
true_tx, true_ty = 3.0, -2.0
c1 = np.array([5, 5])
c2 = np.array([5, -5])

# Calculate rotated positions
rad = np.radians(true_theta_deg)
rot_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

c1_offset = (rot_mat @ c1) + np.array([true_tx, true_ty])
c2_offset = (rot_mat @ c2) + np.array([true_tx, true_ty])

# What the ruler would measure (distances from reference corners)
dist1 = np.linalg.norm(c1_offset - c1)
dist2 = np.linalg.norm(c2_offset - c2)

print(f"--- FORWARD STEP (Measurements) ---")
print(f"Input Rotation: {true_theta_deg}°")
print(f"Input Translation: ({true_tx}, {true_ty})")
print(f"Generated Ruler Measurement d1: {dist1:.4f}")
print(f"Generated Ruler Measurement d2: {dist2:.4f}\n")

# --- INVERSE STEP (Verification) ---
s1, s2 = calculate_plate_offset(true_theta_deg, dist1, dist2, (5, 5), (5, -5))

print(f"--- INVERSE STEP (Recovery) ---")
print(f"Solution A: x={s1[0]:.2f}, y={s1[1]:.2f} (Total Dist: {np.linalg.norm(s1):.2f})")
print(f"Solution B: x={s2[0]:.2f}, y={s2[1]:.2f} (Total Dist: {np.linalg.norm(s2):.2f})")
