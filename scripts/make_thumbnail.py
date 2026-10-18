"""Create the YouTube thumbnail (1280x720 PNG) for the MLOps walkthrough video.

    python scripts/make_thumbnail.py
Edit NAME below if you want your name on it (leave "" for none).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

NAME = "Dinesh Kumar Yadav"                                   # e.g. "Your Name"
OUT = Path(__file__).resolve().parent.parent / "screenshots" / "video_thumbnail.png"

W, H = 1280, 720
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# background gradient (dark navy -> blue)
grad = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(grad, extent=[0, W, 0, H], aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bg", ["#0B1F3A", "#1B4B82"]), zorder=0)

# ECG line across the lower half (heart-rate motif)
x = np.linspace(0, W, 1400)
y = np.full_like(x, 190.0)
for c in (150, 470, 790, 1110):
    y += 80 * np.exp(-((x - c) / 9) ** 2)                # R spike
    y -= 30 * np.exp(-((x - c + 26) / 8) ** 2)           # Q dip
    y -= 42 * np.exp(-((x - c - 24) / 10) ** 2)          # S dip
    y += 22 * np.exp(-((x - c - 95) / 28) ** 2)          # T wave
    y += 14 * np.exp(-((x - c + 90) / 22) ** 2)          # P wave
for lw, a in ((14, 0.10), (8, 0.18), (3.5, 1.0)):
    ax.plot(x, y, color="#FF5A6E", lw=lw, alpha=a, solid_capstyle="round", zorder=2)

# top badge
ax.add_patch(FancyBboxPatch((70, 612), 410, 54, boxstyle="round,pad=0,rounding_size=27",
                            fc="#2EC4B6", ec="none", zorder=3))
ax.text(275, 639, "MLOps ASSIGNMENT 1", ha="center", va="center", fontsize=21, fontweight="bold",
        color="#0B1F3A", zorder=4)

# title
ax.text(70, 520, "Heart Disease", ha="left", va="center", fontsize=80, fontweight="bold",
        color="white", zorder=4)
ax.text(70, 415, "Risk Prediction", ha="left", va="center", fontsize=80, fontweight="bold",
        color="#7FE3D8", zorder=4)
ax.text(72, 330, "End-to-end MLOps pipeline walkthrough", ha="left", va="center", fontsize=30,
        color="#DCE9F9", zorder=4)
if NAME:
    ax.text(72, 285, f"by {NAME}", ha="left", va="center", fontsize=22, color="#9FB6D3", zorder=4)

# tool chips
chips = ["MLflow", "GitHub Actions", "Docker", "Kubernetes", "Grafana"]
x0 = 70
for chip in chips:
    w = 28 + 17.5 * len(chip)
    ax.add_patch(FancyBboxPatch((x0, 36), w, 52, boxstyle="round,pad=0,rounding_size=26",
                                fc="#0B1F3A", ec="#7FE3D8", lw=2, alpha=0.92, zorder=3))
    ax.text(x0 + w / 2, 62, chip, ha="center", va="center", fontsize=19, fontweight="bold",
            color="white", zorder=4)
    x0 += w + 16

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="#0B1F3A")
print("Saved", OUT)
