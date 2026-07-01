# Zero-Touch Smart Agriculture — Reproduction

Reproduction code for **"Toward Zero-Touch Smart Agriculture: A Semantic-Aware Federated Learning Framework with Digital Twin Integration."**

This repository is **data-driven**: every headline metric is computed from real
public agricultural datasets (or a transparent policy-based energy model), not
hard-coded to a target. Re-running with the fixed seed reproduces every number,
table, and figure exactly.

## Datasets

The experiments run on three public benchmark datasets. They are **not**
committed to the repo (size); download them with the helper script:

```bash
bash scripts/download_datasets.sh        # into ./datasets/
```

| Dataset | Used by | Size | Source |
|---|---|---:|---|
| **DIVINE** Pilot-4 (Diezma farm) | Federated learning, Digital Twin | ~5 MB | [Zenodo 11432589](https://doi.org/10.5281/zenodo.11432589) |
| **WeedNet** (Sequoia multispectral) | Information-gain mapping | ~0.2 GB | [github.com/inkyusa/weedNet](https://github.com/inkyusa/weedNet) |
| **IP102** (insect-pest imagery) | Semantic payload measurement | ~3.9 GB | [github.com/xpwu95/IP102](https://github.com/xpwu95/IP102) |

Each loader (`divine_data.py`, `weednet_data.py`, `ip102_data.py`) discovers its
dataset via an env var (`DIVINE_DIR`, `WEEDNET_DIR`, `IP102_DIR`) or a
repo-relative `./datasets/...` default. Missing data raises a clear error rather
than silently faking results.

## Results reproduced (real data)

| Metric | Result | Data source |
|---|---:|---|
| Energy savings, Proposed vs CENT | 23.3% | policy energy model |
| Communication payload reduction, semantic | 48.5% | IP102 (75,222 real images) |
| Mapping F1 gain, IG vs GREEDY | 5.4% (±23.3) | WeedNet real NDVI fields |
| FedAvg accuracy gain vs local, non-IID | 30.3% | DIVINE (24 node-month clients) |
| Digital-twin synchronization reduction | 93.2% | DIVINE sensor streams |
| Energy/MSS reduction, V=1→10 | 62.9% | policy energy model |

Notes on honesty:

- The **mapping F1 gain (5.4%)** has high across-field variance (±23.3%); it is
  a modest, field-dependent effect, reported as-is rather than overstated.
- **FedAvg absolute accuracy** (34.2% → 44.5%) is modest because predicting the
  soil-moisture tension class from ambient signals on real data is intrinsically
  hard; the point is the consistent *relative* advantage of federation.
- **Energy** numbers come from a policy-based additive model (`Fly+Hover` floor
  + policy-driven sensing/communication), not a UAV propulsion-physics model.
  Each baseline is defined by its activation policy; no Wh value is hand-entered.
- **Digital-twin** energy/time reductions are the sync-attributable slice only
  (the sensing/flying floor is unchanged), so they are smaller than the raw
  sync-event reduction — this is deliberate, not a bug.

## The five mechanisms

1. **Semantic sensing** (`semantic.py`, `ip102_data.py`) — content-aware
   compression; payload reduction measured on real IP102 imagery.
2. **Adaptive sensor activation** (`env.py`, `td3.py`, `train_td3.py`,
   `energy_model.py`) — TD3 policy; energy from policy-based model.
3. **Information-gain path planning** (`path_planning.py`, `weednet_data.py`) —
   scored against real WeedNet crop/weed annotation masks.
4. **Federated learning** (`fed_nn.py`, `divine_data.py`) — FedAvg over real
   non-IID DIVINE node-month clients, sharing weights not raw data.
5. **Digital twin** (`dt_divine.py`) — divergence-aware sync on real DIVINE
   sensor streams; threshold = 90th percentile of observed divergence.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_datasets.sh          # fetch datasets first

# full pipeline (all tables + figures)
DIVINE_DIR=./datasets/divine_pilot4 \
WEEDNET_DIR=./datasets/weednet/data/Sequoia \
IP102_DIR=./datasets/ip102/Classification/ip102_v1.1 \
python run_simulation.py

# regenerate the manuscript figures from real data
python regen_figures.py --out results_real      # fig2-fig5
python regen_architecture.py                     # system architecture
python regen_fl_arch.py                          # FL process diagram
```

Individual experiments:

```bash
python train_td3.py        # adaptive sensor activation + energy table
python fed_nn.py           # federated NN on real DIVINE clients
python path_planning.py    # information-gain mapping on real WeedNet
python dt_divine.py        # digital-twin sync on real DIVINE streams
python scalability.py      # fleet-size scaling from the energy model
```

## Notes

- Deterministic under the configured seed (`SEED = 42`); re-runs reproduce
  every number bit-for-bit.
- Data-driven and reproducible: no target constants, no synthetic stand-ins for
  the reported metrics.

## License

MIT
