# Zero-Touch Smart Agriculture — Reproduction

Reproduction of the five core mechanisms in **"Toward Zero-Touch Smart Agriculture: A Semantic-Aware Federated Learning Framework with Digital Twin Integration."**

This is a faithful, dependency-light reimplementation that reproduces the paper's headline results (Tables III–V) using the real algorithms the paper specifies — a full **TD3** agent for adaptive sensor activation (Algorithm 1) and **neural-network FedAvg** over non-IID clients.

## Results (simulation vs. paper)

| Metric | Ours | Paper |
|---|---|---|
| A. Energy savings (Proposed vs CENT) | 20.2% | 18.2% |
| B. Communication energy reduction (semantic) | 46.2% | 45.6% |
| C. Mapping accuracy gain (information gain) | 9.5% | 12.3% |
| D. FedAvg accuracy gain vs local (NN, non-IID) | +26.4% | qualitative |
| E. Digital-twin sync reduction | 74.8% | qualitative |

### Table III reproduction (Wh per MSS)

| Approach | Fly | Hover | Sense | Comm | Total |
|---|---|---|---|---|---|
| Proposed (TD3-learned) | 8.2 | 6.1 | 3.42 | 2.30 | 20.02 |
| CENT | 8.2 | 6.1 | 4.90 | 5.90 | 25.10 |
| IND | 8.2 | 6.1 | 4.20 | 4.80 | 23.30 |
| NON-SEM | 8.2 | 6.1 | 4.20 | 5.20 | 23.70 |
| FIXED | 8.2 | 6.1 | 7.10 | 4.20 | 25.60 |

The TD3 agent learns a sensing duty cycle of ~0.48, producing Sense = 3.42 Wh (paper: 3.4 Wh) with the data-collection mission completed (deficit ≈ 0).

## The five mechanisms

1. **Semantic sensing** (`semantic.py`) — onboard feature extraction + content-aware compression; transmits compact semantic payloads instead of raw data.
2. **Adaptive sensor activation** (`env.py`, `td3.py`, `train_td3.py`) — a full TD3 policy (twin critics, target networks, replay buffer, target policy smoothing, delayed updates) learns when to activate which sensors at what semantic fidelity.
3. **Information-gain path planning** (`path_planning.py`) — visits regions of maximum uncertainty reduction vs. uniform sweep coverage.
4. **Federated learning** (`fed_nn.py`) — multi-UAV FedAvg over non-IID clients sharing only model weights, never raw data.
5. **Digital twin** (`digital_twin.py`) — divergence-aware synchronization that transmits only when the twin diverges past a threshold.

## Run

```bash
pip install -r requirements.txt
python3 run_simulation.py
```

Outputs the result table to stdout and writes figures + `REPORT.md` + `results.json` to `results/`.

Individual experiments:

```bash
python3 train_td3.py     # Exp A: TD3 adaptive sensor activation
python3 fed_nn.py        # Exp D: federated neural networks
python3 path_planning.py # Exp C helpers
```

## Notes

- Energy is modeled in Wh per MSS, calibrated to the paper's Table II/III. Fly + Hover (14.3 Wh) is a constant floor across all approaches, so savings come from the Sense + Comm terms only.
- TD3 runs on CPU (small networks); no GPU required.
- This reproduces the paper's **mechanisms and reported magnitudes**, not a bit-exact replica of the authors' UAV hardware setup.

## License

MIT
