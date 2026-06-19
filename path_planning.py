"""Information-gain path planning (paper Sec. V).

A smooth ground-truth field is reconstructed from noisy point measurements.
Two planners spend the same measurement budget:
  - sweep   : uniform lattice (lawnmower-style full-field coverage baseline)
  - ig_greedy: repeatedly measure the cell of maximum uncertainty (max IG)

Mapping accuracy = 1 - RMSE / field_std. Exp. C compares the two.
"""
import numpy as np
import config as C


def make_field(seed=C.SEED):
    rng = np.random.default_rng(seed)
    M, N = C.M, C.N
    xs, ys = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    field = np.zeros((M, N))
    for _ in range(8):
        cx, cy = rng.uniform(0, M), rng.uniform(0, N)
        amp = rng.uniform(0.5, 1.5)
        w = rng.uniform(2.5, 7.0)
        field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w ** 2))
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    return field


def reconstruct(order, field, seed=0):
    """Kernel-weighted Bayesian reconstruction from measurements in `order`."""
    rng = np.random.default_rng(seed)
    M, N = field.shape
    xs, ys = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    est = np.full((M, N), float(field.mean()))
    prec = np.full((M, N), 1.0)                       # prior precision (var = 1)
    for (cx, cy) in order:
        y = field[cx, cy] + rng.normal(0, C.SIGMA_OBS)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        w = np.exp(-d2 / (2 * C.ELL ** 2))
        gain = w / (C.SIGMA_OBS ** 2)
        new_prec = prec + gain
        est = (est * prec + gain * y) / new_prec
        prec = new_prec
    rmse = float(np.sqrt(np.mean((est - field) ** 2)))
    acc = 1.0 - rmse / (field.std() + 1e-9)
    return acc, est


def sweep_order(M, N, budget):
    """Conventional uniform sweep coverage: a coarse lattice spread evenly over
    the whole field. Same measurement budget as the information-gain planner,
    but allocated blindly rather than toward uncertain regions.
    """
    k = max(1, int(np.sqrt(budget)))
    xs = np.linspace(0, M - 1, k).astype(int)
    ys = np.linspace(0, N - 1, k).astype(int)
    order = [(int(x), int(y)) for x in xs for y in ys]
    return order[:budget]


def ig_order(field, budget):
    """Greedy max-information-gain ordering (variance-reduction surrogate)."""
    M, N = field.shape
    xs, ys = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    var = np.full((M, N), 1.0)
    order = []
    for _ in range(budget):
        idx = np.unravel_index(int(np.argmax(var)), var.shape)
        order.append((int(idx[0]), int(idx[1])))
        d2 = (xs - idx[0]) ** 2 + (ys - idx[1]) ** 2
        w = np.exp(-d2 / (2 * C.ELL ** 2))
        var *= (1.0 - 0.9 * w)                        # IG ~ log(var/(var+sigma^2))
        var = np.clip(var, 1e-4, None)
    return order


def run_mapping_experiment(seed=C.SEED):
    field = make_field(seed)
    M, N = field.shape
    sweep = sweep_order(M, N, C.MEAS_BUDGET)
    ig = ig_order(field, C.MEAS_BUDGET)
    acc_sweep, est_sweep = reconstruct(sweep, field, seed=1)
    acc_ig, est_ig = reconstruct(ig, field, seed=1)
    improvement = (acc_ig - acc_sweep) / max(acc_sweep, 1e-9)
    return {
        "acc_sweep": float(acc_sweep),
        "acc_ig": float(acc_ig),
        "improvement": float(improvement),
        "field": field,
        "est_sweep": est_sweep,
        "est_ig": est_ig,
    }


def run_mapping_experiment_avg(n_seeds=12, base=100):
    """Seed-averaged mapping gain (stable headline number) plus one
    representative field for visualisation."""
    accs_sweep, accs_ig, imps = [], [], []
    for s in range(n_seeds):
        r = run_mapping_experiment(seed=base + s)
        accs_sweep.append(r["acc_sweep"])
        accs_ig.append(r["acc_ig"])
        imps.append(r["improvement"])
    rep = run_mapping_experiment(seed=base)        # field for figure
    return {
        "acc_sweep": float(np.mean(accs_sweep)),
        "acc_ig": float(np.mean(accs_ig)),
        "improvement": float(np.mean(imps)),
        "improvement_std": float(np.std(imps)),
        "field": rep["field"],
        "est_sweep": rep["est_sweep"],
        "est_ig": rep["est_ig"],
    }
