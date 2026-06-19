# Zero-Touch Smart Agriculture - Reproduction Report

Paper-faithful reproduction of the five mechanisms. Exp. A uses a full TD3 agent (Algorithm 1); Exp. D uses neural-network FedAvg over non-IID clients. Numbers are relative-unit reproductions of the paper's mechanisms, not a bit-exact replica of the authors' UAV hardware setup.

| Metric | Our simulation | Paper |
|---|---|---|
| A. Energy savings (TD3 adaptive activation) | 20.2% | 18.2% |
| B. Communication payload reduction | 46.2% | 45.6% |
| C. Mapping accuracy gain (information gain) | 9.5% | 12.3% |
| D. FedAvg accuracy gain vs local (NN, non-IID) | 26.4% | (qualitative) |
| E. Digital-twin sync reduction | 74.8% | (qualitative) |

TD3 learned sensing duty 0.48, gamma 0.66; mission deficit 0.2 (0 = all sensor quotas met).

### Table III reproduction (Wh per MSS)

| Approach | Fly | Hover | Sense | Comm | Total |
|---|---|---|---|---|---|
| Proposed | 8.2 | 6.1 | 3.42 | 2.30 | 20.02 |
| CENT | 8.2 | 6.1 | 4.90 | 5.90 | 25.10 |
| IND | 8.2 | 6.1 | 4.20 | 4.80 | 23.30 |
| NON-SEM | 8.2 | 6.1 | 4.20 | 5.20 | 23.70 |
| FIXED | 8.2 | 6.1 | 7.10 | 4.20 | 25.60 |

## Figures
- `fig1_headline.png` - headline gains vs reported
- `fig2_energy.png` - adaptive activation convergence
- `fig3_mapping.png` - belief-map reconstruction
- `fig4_federated.png` - federated convergence
