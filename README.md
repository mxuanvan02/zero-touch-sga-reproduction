# Zero-Touch Smart Agriculture — Reproduction

Reproduction code for **"Toward Zero-Touch Smart Agriculture: A Semantic-Aware Federated Learning Framework with Digital Twin Integration."**

This repository is intentionally **paper-faithful**: it reproduces the manuscript's reported tables/figures using the same modeling convention used in the paper. In particular, the energy experiment reports **mission energy in Wh per MSS** with a fixed Fly+Hover floor shared by all baselines; the optimization gain comes from adaptive sensing and semantic communication.

## Results reproduced by `run_simulation.py`

| Metric | Reproduced | Manuscript target |
|---|---:|---:|
| Energy savings, Proposed vs CENT | 18.1% | 18.2% |
| Communication energy reduction, semantic | 46.2% | 45.6% |
| Mapping F1 gain, IG vs GREEDY | 13.5% | 12.3% |
| FedAvg accuracy gain vs local, non-IID | 26.4% | qualitative |
| Digital-twin synchronization reduction | 73.8% | qualitative |
| Energy/MSS reduction, V=1→10 | 35.5% | 35.6% |

### Energy table convention

The manuscript's energy table is reproduced as:

| Approach | Fly | Hover | Sense | Comm | Total |
|---|---:|---:|---:|---:|---:|
| Proposed | 8.2 | 6.1 | 3.42 | 2.83 | 20.55 |
| CENT | 8.2 | 6.1 | 4.90 | 5.90 | 25.10 |
| IND | 8.2 | 6.1 | 4.20 | 4.80 | 23.30 |
| NON-SEM | 8.2 | 6.1 | 4.20 | 5.20 | 23.70 |
| FIXED | 8.2 | 6.1 | 7.10 | 4.20 | 25.60 |

`Fly + Hover = 14.3 Wh` is a constant mission floor across approaches. Therefore, the 18.1% total-energy saving is caused by the controllable terms: sensing duty and semantic communication.

This repository does **not** replace the manuscript's table with a propulsion-physics UAV model. A physical propulsion model would be a different experiment and would change the denominator of the energy-saving claim.

## The five mechanisms

1. **Semantic sensing** (`semantic.py`) — content-aware compression; transmits compact semantic payloads instead of raw data.
2. **Adaptive sensor activation** (`env.py`, `td3.py`, `train_td3.py`) — TD3 policy with twin critics, target networks, replay buffer, target policy smoothing, and delayed updates.
3. **Information-gain path planning** (`path_planning.py`) — uncertainty-reduction path planning compared against greedy/sweep coverage.
4. **Federated learning** (`fed_nn.py`) — neural-network FedAvg over non-IID clients, sharing model weights rather than raw data.
5. **Digital twin** (`digital_twin.py`) — divergence-aware synchronization and DT-impact metrics for energy, mission time, revisit rate, and synchronization events.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_simulation.py
```

Outputs are written to `results/`:

- `results/results.json` — machine-readable metrics
- `results/REPORT.md` — markdown summary and reproduced tables
- `results/fig1_headline.png` ... `results/fig5_scalability.png` — manuscript figures

Individual experiments:

```bash
python train_td3.py      # adaptive sensor activation and energy table
python fed_nn.py         # federated neural network experiment
python path_planning.py  # information-gain mapping helpers
```

## Notes

- The code is deterministic under the configured seed (`SEED = 42`).
- The reproduction is mechanism-faithful and table-faithful to the manuscript, not a bit-exact replica of a specific UAV hardware platform.
- Energy units follow the manuscript's Wh-per-MSS reporting convention.

## License

MIT
