"""Energy model + baselines (per-MSS mission energy, Wh).

Design contract (project rule): NOTHING is a hand-typed result. Every
Sense/Comm/Total number is COMPUTED from physical coefficients (config.py) and
each baseline's *behaviour policy* (duty cycle, whether it transmits semantic or
raw payloads, and how often it transmits). Change a coefficient or a policy and
every number moves accordingly -- so the table is a model output, not a target.

Model
    E_total = E_fly_hover (const floor) + E_sense + E_comm
    E_sense = SENSE_FULL * duty                       (SENSE_FULL = sum(SENSOR_ENERGY)*T)
    E_comm  = COMM_FULL  * comm_duty * (rho(gamma) if semantic else 1)
              (COMM_FULL = KC_RAW * T)
The "Proposed" row uses the TD3-learned duty (runtime), semantic comm at the
learned gamma, and the same comm model as every baseline.
"""
import numpy as np
import config as C
from semantic import compression_ratio

FLY_HOVER = C.E_FLY_HOVER                       # constant floor, identical for all
SENSE_FULL = float(C.SENSOR_ENERGY.sum()) * C.T  # full-activation sensing energy
COMM_FULL = C.KC_RAW * C.T                        # full raw-transmission comm energy


def sense_energy(duty):
    """Sensing energy for an average activation duty cycle in [0,1]."""
    return SENSE_FULL * float(duty)


def comm_energy(comm_duty, semantic, gamma):
    """Communication energy: raw baseline scaled by transmit duty, and by the
    semantic compression ratio rho(gamma) when semantic encoding is used."""
    e = COMM_FULL * float(comm_duty)
    if semantic:
        e *= compression_ratio(gamma)
    return e


def _row(duty, comm_duty, semantic, gamma):
    s = sense_energy(duty)
    c = comm_energy(comm_duty, semantic, gamma)
    return {"sense": s, "comm": c, "fly_hover": FLY_HOVER,
            "total": FLY_HOVER + s + c}


def baselines(proposed_duty, proposed_gamma=None, proposed_comm_duty=1.0):
    """Compute the energy breakdown for Proposed + every baseline policy.

    proposed_duty / proposed_gamma come from the TD3 run (train_td3.py); the
    baseline policies come from config.BASELINE_POLICIES. Nothing is typed by
    hand -- each cell is _row(...) evaluated on a policy.
    """
    gamma = C.GAMMA[0] if proposed_gamma is None else float(proposed_gamma)
    rows = {"Proposed": _row(proposed_duty, proposed_comm_duty, True, gamma)}
    for name, p in C.BASELINE_POLICIES.items():
        rows[name] = _row(p["duty"], p["comm_duty"], p["semantic"], p["gamma"])
    return rows


def headline_savings(rows, ref="CENT"):
    """Relative total-energy saving of Proposed vs a reference baseline.
    Computed from the model outputs -- not asserted."""
    return 1.0 - rows["Proposed"]["total"] / rows[ref]["total"]


if __name__ == "__main__":
    # Demo with a representative learned duty; real value comes from train_td3.
    rows = baselines(proposed_duty=0.48, proposed_gamma=0.66)
    print(f"{'Approach':<10}{'Sense':>7}{'Comm':>7}{'Total':>8}")
    for name, r in rows.items():
        print(f"{name:<10}{r['sense']:>7.2f}{r['comm']:>7.2f}{r['total']:>8.2f}")
    print(f"Saving vs CENT: {100*headline_savings(rows):.1f}%")
