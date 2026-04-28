import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(__file__), "Webcam Camera Calibration - Current.csv")
df = pd.read_csv(csv_path)

grouped = df.groupby("Measuring Tape [cm]")
tape = grouped["Measuring Tape [cm]"].mean().index.values
calibrated_mean = grouped["Camera calibrated [cm]"].mean()
calibrated_std = grouped["Camera calibrated [cm]"].std()
naive_mean = grouped["Naive camera calibration [cm]"].mean()
naive_std = grouped["Naive camera calibration [cm]"].std()

fig, ax = plt.subplots(figsize=(3.5, 4), dpi=150)

# Scatter with error bars 
ax.errorbar(tape, calibrated_mean, yerr=calibrated_std, fmt="o", capsize=5, capthick=1.5, elinewidth=1.5, color="C0")
ax.errorbar(tape, naive_mean, yerr=naive_std, fmt="s", capsize=5, capthick=1.5, elinewidth=1.5, color="C1")

# Best fit lines 
cal_slope, cal_intercept = np.polyfit(tape, calibrated_mean, 1)
naive_slope, naive_intercept = np.polyfit(tape, naive_mean, 1)

x_fit = np.linspace(tape.min(), tape.max(), 200)
cal_std_mean = calibrated_std.mean()
naive_std_mean = naive_std.mean()
ax.plot(x_fit, cal_slope * x_fit + cal_intercept, "-", color="C0",
        label=f"Calibrated (slope={cal_slope:.2f}, \n stdev={cal_std_mean:.2f} cm)")
ax.plot(x_fit, naive_slope * x_fit + naive_intercept, "--", color="C1",
        label=f"Uncalibrated (slope={naive_slope:.2f}, \n stdev={naive_std_mean:.2f} cm)")
ax.plot(x_fit, x_fit, "k:", label="Ideal (slope=1)")

ax.set_xlabel("Measuring Tape [cm]", fontsize=12)
ax.set_ylabel("Camera Measurement [cm]", fontsize=12)
ax.xaxis.set_major_locator(plt.MultipleLocator(5))

ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=1)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "CameraCharacterization.png")
plt.savefig(output_path, bbox_inches="tight")
plt.show()
