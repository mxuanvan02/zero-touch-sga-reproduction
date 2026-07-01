"""Regenerate the system-architecture figure (Fig. 1) with correct data flow.

The previous static PNG had three data-flow errors:
  1. "Control commands" pointed Sensors -> Server (backwards; commands go down).
  2. "A2G Semantic Upload" started from the Edge Node, which has no Semantic
     Encoder; only the Mobile Sensors (MSS) encode, so the uplink must start
     there.
  3. The Central Server had no downlink at all (no global-model / command
     distribution), despite being the aggregator + Digital Twin.

This script redraws the diagram with a correct, legible layout:
  * Mobile Sensors --[A2G Semantic Upload]--> Central Server   (uplink)
  * Central Server --[Global model + Control]--> Edge Node     (downlink)
  * Edge Node <--[sensing data / activation]--> Mobile Sensors (labelled)

Deterministic and reproducible: run `python regen_architecture.py`.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ARCH_OUT", os.path.join(HERE, "results_real"))
os.makedirs(OUT, exist_ok=True)

NAVY = "#1f2d5a"
TEAL = "#1a8a7a"
BLUE = "#2a6fb0"
RED = "#c0392b"
GREEN = "#27ae60"
GREY = "#555555"


def _box(ax, xy, w, h, color, title, sub, lines):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0, facecolor=color, zorder=2))
    cx = x + w / 2
    # vertical layout with generous spacing so title/sub/body never crowd
    ax.text(cx, y + h - 0.26, title, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)
    ax.text(cx, y + h - 0.60, sub, ha="center", va="center",
            color="#dfe6f2", fontsize=8.5, style="italic", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(cx, y + h - 0.95 - i * 0.30, ln, ha="center", va="center",
                color="#eef2f9", fontsize=8, zorder=3)


def _arrow(ax, p0, p1, color, label, lab_xy, rad=0.0, lab_color="black"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=18, linewidth=2.0,
        color=color, zorder=1))
    if label:
        ax.text(lab_xy[0], lab_xy[1], label, ha="center", va="center",
                fontsize=8.5, color=lab_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.2)
    ax.axis("off")

    # --- boxes (taller so 4 text rows never crowd) ---
    # Central server (top center)
    _box(ax, (3.2, 5.0), 3.6, 1.9, NAVY,
         "CENTRAL SERVER", "(Digital Twin & Aggregator)",
         ["FedAvgM global model", "Event-triggered sync"])
    # Edge node (bottom left)
    _box(ax, (0.3, 0.9), 3.4, 1.9, TEAL,
         "EDGE NODE", "(UAV Controller)",
         ["TD3 DRL policy", "Sensing decision"])
    # Mobile sensors (bottom right)
    _box(ax, (6.3, 0.9), 3.4, 1.9, BLUE,
         "MOBILE SENSORS", "(MSS / UAV Swarm)",
         ["Semantic encoder", "Path planning (IG)"])

    # --- arrows (corrected data flow) ---
    # Uplink: Mobile Sensors -> Server  (A2G semantic upload; MSS has encoder)
    _arrow(ax, (8.0, 2.8), (6.2, 5.0), RED,
           "A2G semantic\nupload", (7.85, 4.0), rad=-0.16)
    # Downlink: Server -> Edge Node (global model + control commands)
    _arrow(ax, (3.8, 5.0), (2.0, 2.8), GREEN,
           "Global model +\ncontrol commands", (2.15, 4.0), rad=-0.16)
    # Downlink: Server -> Mobile Sensors (global model broadcast to swarm too)
    _arrow(ax, (5.4, 5.0), (7.4, 2.8), GREEN, "", (0, 0), rad=0.16)
    # Edge <-> Sensors: sensing/activation exchange (bidirectional, labelled)
    _arrow(ax, (3.7, 1.5), (6.3, 1.5), GREY, "", (0, 0), rad=0.0)
    _arrow(ax, (6.3, 1.9), (3.7, 1.9), GREY, "", (0, 0), rad=0.0)
    ax.text(5.0, 2.28, "sensing data / activation",
            ha="center", va="center", fontsize=8.5, color=GREY,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            zorder=4)

    fig.tight_layout()
    out = os.path.join(OUT, "system_architecture.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("ARCH_OK ->", out)


if __name__ == "__main__":
    main()
