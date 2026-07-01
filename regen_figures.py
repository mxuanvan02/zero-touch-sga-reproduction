"""Regenerate manuscript figures fig2-fig5 from the REAL data-driven pipeline.

Every number/label on every figure is pulled live from the grounded modules:
  * fig2 (energy)      <- energy_model policy baselines (real coefficients)
  * fig3 (mapping)     <- path_planning on real WeedNet NDVI field + mask
  * fig4 (federated)   <- fed_nn on real DIVINE non-IID clients
  * fig5 (scalability) <- scalability from the policy energy model

No target/paper constants, no synthetic 99.7%/20.5-Wh labels. Re-running with
the same seed + data reproduces the figures exactly.

Usage (env vars must point at the datasets on HDD):
  DIVINE_DIR=... WEEDNET_DIR=... IP102_DIR=... python regen_figures.py --out <dir>
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from energy_model import baselines, headline_savings
from train_td3 import train_td3
from path_planning import run_mapping_experiment_avg
from fed_nn import run_federated_nn
from scalability import scalability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir for figures")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "results_real")
    os.makedirs(out, exist_ok=True)

    # ---------------- gather REAL numbers ----------------
    print("TD3 energy (policy model)...")
    energy = train_td3(verbose=False)
    rows = baselines(energy["proposed_duty"])
    prop_total = rows["Proposed"]["total"]
    cent_total = rows["CENT"]["total"]
    savings = headline_savings(rows)

    print("Mapping (real WeedNet NDVI)...")
    mapping = run_mapping_experiment_avg()          # MAP_SOURCE=weednet via env

    print("Federated (real DIVINE)...")
    fed = run_federated_nn()                        # FL_SOURCE=divine via env

    print("Scalability (policy model)...")
    scal_rows, scal_red = scalability((1, 3, 5, 10))

    # ---------------- fig2: energy convergence ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energy["energy_curve"], color="#2a7ab9", lw=1.5,
            label="TD3 proposed (per episode)")
    ax.axhline(cent_total, color="#c0392b", ls="--",
               label=f"CENT baseline ({cent_total:.1f} Wh)")
    ax.axhline(prop_total, color="#27ae60", ls=":",
               label=f"Proposed final ({prop_total:.1f} Wh)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mission energy (Wh)")
    ax.set_title(f"TD3 adaptive sensor activation: "
                 f"{100*savings:.1f}% energy saving vs CENT")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_energy.png"), dpi=130); plt.close(fig)

    # ---------------- fig3: mapping fields (real) ----------------
    # All three panels share the same real WeedNet field frame and the same
    # colour scale. Panels 2-3 are the belief maps RECONSTRUCTED from a limited
    # measurement budget, so they are legitimately sparser than the dense
    # ground truth; titles say so explicitly to avoid the "looks broken" read.
    field = mapping["field"]
    est_greedy = mapping["est_sweep"]
    est_ig = mapping["est_ig"]
    vmin = 0.0
    vmax = float(max(field.max(), est_greedy.max(), est_ig.max(), 1e-6))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axs,
        [field, est_greedy, est_ig],
        ["Ground truth\n(real WeedNet NDVI field)",
         f"GREEDY belief map\n(reconstructed, F1={mapping['acc_sweep']:.3f})",
         f"Proposed IG belief map\n(reconstructed, F1={mapping['acc_ig']:.3f})"],
    ):
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="bilinear", aspect="equal")
        ax.set_title(title, fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.025, label="Detection belief")
    fig.savefig(os.path.join(out, "fig3_mapping.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- fig4: federated convergence (real) ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([100*c for c in fed["curve"]], color="#27ae60", lw=2,
            label="FedAvg (global test acc)")
    ax.axhline(100*fed["local_acc"], color="#c0392b", ls="--",
               label=f"Mean local-only ({100*fed['local_acc']:.1f}%)")
    ax.set_xlabel("FL round"); ax.set_ylabel("Global test accuracy (%)")
    ax.set_title(f"FedAvg on real DIVINE non-IID data: "
                 f"{100*fed['local_acc']:.1f}% -> {100*fed['fed_acc']:.1f}% "
                 f"(+{100*fed['improvement']:.1f}%)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_federated.png"), dpi=130); plt.close(fig)

    # ---------------- fig5: scalability (real policy model) ----------------
    Vs = [r["V"] for r in scal_rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.2))
    axA.plot(Vs, [r["energy_per_uav"] for r in scal_rows], "o-",
             color="#2a7ab9", lw=2, label="Energy per UAV")
    axA.set_xlabel("Number of UAVs (V)")
    axA.set_ylabel("Energy per UAV (Wh)", color="#2a7ab9")
    axA.tick_params(axis="y", labelcolor="#2a7ab9")
    axA.set_title(f"Energy/UAV decreases with fleet size "
                  f"({100*scal_red:.1f}% drop V=1->10)")
    axA2 = axA.twinx()
    axA2.plot(Vs, [r["mission_time"] for r in scal_rows], "s--",
              color="#e67e22", lw=1.5, label="Mission time")
    axA2.set_ylabel("Mission time (min)", color="#e67e22")
    axA2.tick_params(axis="y", labelcolor="#e67e22")
    lines = axA.get_lines() + axA2.get_lines()
    axA.legend(lines, [l.get_label() for l in lines], loc="upper right")

    axB.plot(Vs, [r["comm_fl"] for r in scal_rows], "o-", color="#27ae60",
             lw=2, label="FL  O(V)")
    axB.plot(Vs, [r["comm_cent"] for r in scal_rows], "s-", color="#c0392b",
             lw=2, label="CENT  O(V^2)")
    axB.set_xlabel("Number of UAVs (V)")
    axB.set_ylabel("Communication overhead")
    axB.set_title("Communication overhead: FL O(V) vs CENT O(V^2)")
    axB.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_scalability.png"), dpi=130)
    plt.close(fig)

    print("REGEN_OK")
    print(f"savings={100*savings:.1f}% prop_total={prop_total:.1f} cent={cent_total:.1f}")
    print(f"mapping f1: greedy={mapping['acc_sweep']:.3f} ig={mapping['acc_ig']:.3f}")
    print(f"fed: local={100*fed['local_acc']:.1f}% fed={100*fed['fed_acc']:.1f}%")
    print(f"scal_reduction={100*scal_red:.1f}%")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
