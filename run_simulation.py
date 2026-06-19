"""Main runner: executes all five experiments, prints a result table, and
writes figures + a JSON/Markdown report under results/.

Run:  python3 run_simulation.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from semantic import payload_reduction
from train_td3 import train_td3
from path_planning import run_mapping_experiment_avg
from fed_nn import run_federated_nn
from digital_twin import run_digital_twin_experiment
from scalability import scalability

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def pct(x):
    return f"{100*x:.1f}%"


def main():
    np.random.seed(C.SEED)

    # ---- Exp. B: semantic payload reduction ----
    red, raw, sem, payload_red, table4 = payload_reduction()
    # ---- Exp. A: TD3 adaptive sensor activation (Algorithm 1) ----
    print("Training TD3 adaptive sensor-activation policy...")
    energy = train_td3(verbose=False)
    # ---- Exp. C: information-gain mapping accuracy (seed-averaged) ----
    mapping = run_mapping_experiment_avg()
    # ---- Exp. D: neural FedAvg over non-IID clients ----
    print("Training federated neural networks...")
    fed = run_federated_nn()
    # ---- Exp. E: digital twin sync reduction ----
    dt = run_digital_twin_experiment()
    # ---- Exp. F: scalability analysis V=1,3,5,10 (Sec. IX.G, Fig. 9) ----
    scal_rows, scal_reduction = scalability((1, 3, 5, 10))

    results = {
        "energy_savings": energy["savings"],
        "payload_reduction": red,
        "mapping_improvement": mapping["improvement"],
        "fl_improvement": fed["improvement"],
        "dt_sync_reduction": dt["sync_reduction"],
        "detail": {
            "energy": {"proposed_total": energy["proposed_total"],
                       "cent_total": energy["cent_total"],
                       "proposed_duty": energy["proposed_duty"],
                       "mean_gamma": energy["mean_gamma"],
                       "savings": energy["savings"],
                       "mean_deficit": energy["mean_deficit"],
                       "table3": energy["table3"]},
            "mapping": {k: mapping[k] for k in ("acc_sweep", "acc_ig", "improvement")},
            "federated": {"local_acc": fed["local_acc"], "fed_acc": fed["fed_acc"],
                          "improvement": fed["improvement"]},
            "digital_twin": dt,
            "scalability": {"rows": scal_rows, "energy_reduction_1to10": scal_reduction},
            "paper_targets": {
                "energy_savings": 0.182,
                "payload_reduction": 0.456,
                "mapping_improvement": 0.123,
            },
        },
    }

    # ---------- console table ----------
    print("\n" + "=" * 64)
    print(" Zero-Touch Smart Agriculture - Simulation Results (paper-faithful)")
    print("=" * 64)
    print(f"{'Metric':<36}{'Ours':>9}{'Paper':>9}")
    print("-" * 64)
    print(f"{'A. Energy savings (Proposed vs CENT)':<36}{pct(energy['savings']):>9}{'18.2%':>9}")
    print(f"{'B. Comm. energy reduction (semantic)':<36}{pct(red):>9}{'45.6%':>9}")
    print(f"{'C. Mapping F1 gain (IG vs GREEDY)':<36}{pct(mapping['improvement']):>9}{'12.3%':>9}")
    print(f"{'D. FedAvg acc gain vs local (NN)':<36}{pct(fed['improvement']):>9}{'-':>9}")
    print(f"{'E. Digital-twin sync reduction':<36}{pct(dt['sync_reduction']):>9}{'-':>9}")
    print(f"{'F. Energy/UAV drop V=1->10 (scaling)':<36}{pct(scal_reduction):>9}{'35.6%':>9}")
    print("-" * 64)
    print(f"  (TD3 mission deficit at eval: {energy['mean_deficit']:.1f}; 0 = complete)")
    print("=" * 64)

    # ---------- figures ----------
    # Fig 1: headline metrics vs paper targets
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Energy\nsavings", "Payload\nreduction", "Mapping\naccuracy gain"]
    ours = [energy["savings"], red, mapping["improvement"]]
    paper = [0.182, 0.456, 0.123]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, [100*v for v in ours], w, label="Our simulation", color="#2a7ab9")
    ax.bar(x + w/2, [100*v for v in paper], w, label="Paper (reported)", color="#f2a154")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Headline gains: simulation vs. reported")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    for i, v in enumerate(ours):
        ax.text(i - w/2, 100*v + 0.5, f"{100*v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate(paper):
        ax.text(i + w/2, 100*v + 0.5, f"{100*v:.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_headline.png"), dpi=130)
    plt.close(fig)

    # Fig 2: TD3 energy convergence (per-episode mission energy)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energy["energy_curve"], color="#2a7ab9", lw=1.5, label="TD3 proposed (per episode)")
    ax.axhline(energy["cent_total"], color="#c0392b", ls="--", label="CENT baseline (25.1 Wh)")
    ax.axhline(energy["proposed_total"], color="#27ae60", ls=":", label=f"Proposed final ({energy['proposed_total']:.1f} Wh)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mission energy (Wh)")
    ax.set_title("TD3 adaptive sensor activation converges toward paper's 20.5 Wh")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_energy.png"), dpi=130); plt.close(fig)

    # Fig 3: mapping fields
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axs,
        [mapping["field"], mapping["est_sweep"], mapping["est_ig"]],
        ["Ground truth",
         f"GREEDY detect (F1={mapping['acc_sweep']:.3f})",
         f"Proposed IG (F1={mapping['acc_ig']:.3f})"],
    ):
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.025)
    fig.savefig(os.path.join(OUT, "fig3_mapping.png"), dpi=130); plt.close(fig)

    # Fig 4: federated convergence (accuracy)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([100*c for c in fed["curve"]], color="#27ae60", lw=2, label="FedAvg (global test acc)")
    ax.axhline(100*fed["local_acc"], color="#c0392b", ls="--", label="Mean local-only")
    ax.set_xlabel("FL round"); ax.set_ylabel("Global test accuracy (%)")
    ax.set_title("Federated aggregation beats isolated local training (non-IID)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_federated.png"), dpi=130); plt.close(fig)

    # Fig 5: scalability (Sec. IX.G, Fig. 9) - energy/UAV + comm scaling
    Vs = [r["V"] for r in scal_rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.2))
    axA.plot(Vs, [r["energy_per_uav"] for r in scal_rows], "o-",
             color="#2a7ab9", lw=2, label="Energy per UAV")
    axA.set_xlabel("Number of UAVs (V)"); axA.set_ylabel("Energy per UAV (Wh)", color="#2a7ab9")
    axA.tick_params(axis="y", labelcolor="#2a7ab9")
    axA.set_title(f"Energy/UAV decreases with fleet size ({100*scal_reduction:.1f}% drop V=1->10)")
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
    axB.set_xlabel("Number of UAVs (V)"); axB.set_ylabel("Communication overhead")
    axB.set_title("Communication overhead: FL O(V) vs CENT O(V^2)")
    axB.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_scalability.png"), dpi=130)
    plt.close(fig)

    # ---------- reports ----------
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("# Zero-Touch Smart Agriculture - Reproduction Report\n\n")
        f.write("Paper-faithful reproduction of the five mechanisms. Exp. A uses a "
                "full TD3 agent (Algorithm 1); Exp. D uses neural-network FedAvg "
                "over non-IID clients. Numbers are relative-unit reproductions of "
                "the paper's mechanisms, not a bit-exact replica of the authors' "
                "UAV hardware setup.\n\n")
        f.write("| Metric | Our simulation | Paper |\n|---|---|---|\n")
        f.write(f"| A. Energy savings (TD3 adaptive activation) | {pct(energy['savings'])} | 18.2% |\n")
        f.write(f"| B. Communication payload reduction | {pct(red)} | 45.6% |\n")
        f.write(f"| C. Mapping F1 gain (information gain vs GREEDY) | {pct(mapping['improvement'])} | 12.3% |\n")
        f.write(f"| D. FedAvg accuracy gain vs local (NN, non-IID) | {pct(fed['improvement'])} | (qualitative) |\n")
        f.write(f"| E. Digital-twin sync reduction | {pct(dt['sync_reduction'])} | (qualitative) |\n")
        f.write(f"| F. Energy/UAV reduction (V=1->10, scalability) | {pct(scal_reduction)} | 35.6% |\n\n")
        f.write(f"TD3 learned sensing duty {energy['proposed_duty']:.2f}, gamma "
                f"{energy['mean_gamma']:.2f}; mission deficit {energy['mean_deficit']:.1f} "
                f"(0 = all sensor quotas met).\n\n")
        f.write("### Table III reproduction (Wh per MSS)\n\n")
        f.write("| Approach | Fly | Hover | Sense | Comm | Total |\n|---|---|---|---|---|---|\n")
        for name, row in energy["table3"].items():
            f.write(f"| {name} | 8.2 | 6.1 | {row['sense']:.2f} | {row['comm']:.2f} | {row['total']:.2f} |\n")
        f.write("\n")
        f.write("### Scalability (Sec. IX.G, Fig. 9)\n\n")
        f.write("| V | Energy/UAV (Wh) | Mission time (min) | Comm FL O(V) | Comm CENT O(V^2) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in scal_rows:
            f.write(f"| {r['V']} | {r['energy_per_uav']:.2f} | {r['mission_time']:.2f} "
                    f"| {r['comm_fl']:.2f} | {r['comm_cent']:.2f} |\n")
        f.write("\n")
        f.write("## Figures\n")
        f.write("- `fig1_headline.png` - headline gains vs reported\n")
        f.write("- `fig2_energy.png` - adaptive activation convergence\n")
        f.write("- `fig3_mapping.png` - detection F1 (IG vs GREEDY)\n")
        f.write("- `fig4_federated.png` - federated convergence\n")
        f.write("- `fig5_scalability.png` - scalability with fleet size\n")

    print(f"\nFigures + report written to: {OUT}")
    return results


if __name__ == "__main__":
    main()
