"""Regenerate the federated-learning architecture figure (fl_architecture.png).

The previous static PNG had two rendering defects:
  1. The edge labels ("theta_i upload") were struck through by the arrow lines,
     partially obscuring the text.
  2. The centre client (UAV 2) was missing its dashed red upload arrow, so the
     three columns were visually asymmetric.

This redraw fixes both: the up/down arrows for every client are offset
horizontally so they never sit under the labels, all three clients have a
symmetric (green down-link + dashed red up-link) pair, and each label carries a
white halo box so no arrow can cut through it.

Deterministic and reproducible: run `python regen_fl_arch.py`.
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
GREEN = "#27ae60"
RED = "#c0392b"


def main():
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5.0, 6.6, "FEDERATED LEARNING PROCESS", ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY)

    # --- central server (ellipse-like rounded box, top center) ---
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((5.0, 5.4), 3.6, 1.0, facecolor=NAVY, zorder=2))
    ax.text(5.0, 5.4, "Global Model Aggregation", ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold", zorder=3)

    # --- three client boxes (bottom row) ---
    client_cx = [2.0, 5.0, 8.0]
    box_w, box_h, box_y = 2.4, 1.2, 0.5
    for i, cx in enumerate(client_cx, start=1):
        x = cx - box_w / 2
        ax.add_patch(FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0, facecolor=TEAL, zorder=2))
        ax.text(cx, box_y + box_h - 0.36, f"UAV {i} Client", ha="center",
                va="center", color="white", fontsize=10, fontweight="bold",
                zorder=3)
        ax.text(cx, box_y + 0.34, r"Local update $\theta_%d$" % i, ha="center",
                va="center", color="#eef2f9", fontsize=9, zorder=3)

    server_bottom = 4.9
    box_top = box_y + box_h  # 1.7

    for i, cx in enumerate(client_cx, start=1):
        # downlink (green solid): server -> client, offset slightly left
        _dx = 0.28
        ax.add_patch(FancyArrowPatch(
            (cx - _dx, server_bottom), (cx - _dx, box_top),
            arrowstyle="-|>", mutation_scale=15, linewidth=2.0,
            color=GREEN, zorder=1))
        # uplink (dashed red): client -> server, offset slightly right
        ax.add_patch(FancyArrowPatch(
            (cx + _dx, box_top), (cx + _dx, server_bottom),
            arrowstyle="-|>", mutation_scale=15, linewidth=2.0,
            color=RED, linestyle="--", zorder=1))
        # label with white halo, placed to the SIDE of the arrow pair
        ax.text(cx + 1.05, (server_bottom + box_top) / 2,
                r"$\theta_%d$ upload" % i, ha="center", va="center",
                fontsize=9, color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.95), zorder=4)

    # legend for the two arrow types (upper-left corner, clear of the title,
    # the server ellipse, and every theta-upload label which sit at y~3.3)
    ax.add_patch(FancyArrowPatch((0.3, 6.4), (1.0, 6.4), arrowstyle="-|>",
                 mutation_scale=13, linewidth=2.0, color=GREEN, zorder=3))
    ax.text(1.15, 6.4, "global model (down)", ha="left", va="center",
            fontsize=8.5, color="#444444", zorder=3)
    ax.add_patch(FancyArrowPatch((0.3, 6.0), (1.0, 6.0), arrowstyle="-|>",
                 mutation_scale=13, linewidth=2.0, color=RED,
                 linestyle="--", zorder=3))
    ax.text(1.15, 6.0, "weight upload (up)", ha="left", va="center",
            fontsize=8.5, color="#444444", zorder=3)

    fig.tight_layout()
    out = os.path.join(OUT, "fl_architecture.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("FL_ARCH_OK ->", out)


if __name__ == "__main__":
    main()
