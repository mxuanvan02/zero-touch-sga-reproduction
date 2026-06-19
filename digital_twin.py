"""Digital Twin integration (paper Sec. VII).

Divergence-aware synchronisation: the twin predicts MSS state; a sync
(costly transmission) fires only when divergence d(t) = ||s - s_hat|| exceeds
threshold delta. Compares against periodic sync that transmits every slot.

Demonstrates redundant-transmission / data-collection reduction while keeping
mean divergence bounded.
"""
import numpy as np
import config as C


def run_digital_twin_experiment(seed=C.SEED):
    rng = np.random.default_rng(seed)
    T, V = C.T, C.V

    periodic_syncs = T * V          # transmit every slot for every MSS
    div_periodic = []
    div_adaptive = []
    adaptive_syncs = 0

    for v in range(V):
        phys = rng.normal(0, 1, size=3)
        twin = phys.copy()
        for t in range(T):
            phys = phys + rng.normal(0, C.DT_DRIFT, size=3)   # physical drift
            # periodic: twin resynced every slot -> divergence ~ one drift step
            div_periodic.append(float(np.linalg.norm(rng.normal(0, C.DT_DRIFT, 3))))
            # adaptive: twin coasts on prediction until divergence exceeds delta
            d = float(np.linalg.norm(phys - twin))
            div_adaptive.append(d)
            if d > C.DT_DELTA:
                twin = phys.copy()                             # sync (transmit)
                adaptive_syncs += 1

    reduction = 1.0 - adaptive_syncs / periodic_syncs
    return {
        "periodic_syncs": int(periodic_syncs),
        "adaptive_syncs": int(adaptive_syncs),
        "sync_reduction": float(reduction),
        "mean_div_periodic": float(np.mean(div_periodic)),
        "mean_div_adaptive": float(np.mean(div_adaptive)),
    }
