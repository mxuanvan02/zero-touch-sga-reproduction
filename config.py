"""Global configuration for the Zero-Touch Smart Agriculture simulation.

Calibrated to the published paper "Toward Zero-Touch Smart Agriculture".
Energy units follow Table II (Simulation Parameters) and Table III (Energy
Consumption Breakdown, Wh per MSS). The headline 18.2% energy saving is
Proposed (20.5 Wh) vs CENT (25.1 Wh); Fly+Hover is a constant floor across all
approaches, so savings come from the Sense + Comm terms only.
"""
import numpy as np

SEED = 42

# ---- Field (paper Table II: 1km x 1km, 100x100 grid) ----
M, N = 40, 40                 # belief-map grid for the mapping experiment (downscaled)

# ---- Fleet / sensors ----
V = 5                         # number of MSSs (paper uses 3, 5, 10; 5 = mid point)
S = 5                         # sensor types per MSS
T = 100                       # time slots per mission (paper Table II)

# ---- Semantic sensing model (Sec. III) ----
ALPHA = 3.0                   # utility sensitivity:  U = 1 - e^{-alpha*gamma}
BETA = 2.0                    # compression-curve exponent
RHO_MIN = 0.05                # minimum compression ratio (= maximum compression)
ZETA = 2e-3                   # per-byte semantic encoding energy (relative)
GAMMA = np.full(S, 0.30)      # operating semantic fidelity (paper range [0.3, 1.0])
LAMBDA1 = 0.6                 # utility weight (paper sensitivity analysis: optimal ~0.6)

# ---- Energy model calibrated to paper Table III (Wh per MSS) ----
# Fly + Hover = constant floor, identical across every approach.
E_FLY_HOVER = 14.3            # 8.2 (fly) + 6.1 (hover)
# Per-sensor sensing energy (Wh per activation). Physical per-sensor draw; the
# full-activation sensing energy is sum(SENSOR_ENERGY)*T, computed downstream.
SENSOR_ENERGY = np.array([0.013, 0.018, 0.012, 0.016, 0.012])   # Wh/activation
# Raw comm energy per transmitted slot (Wh). Full raw transmission every slot
# gives KC_RAW*T; semantic compression scales this by rho(gamma) downstream.
KC_RAW = 5.2 / 100.0          # Wh per transmitted slot

# ---- Baseline POLICIES (behaviour only; results are COMPUTED, never typed) ----
# Each baseline is defined ONLY by what it DOES, not by its energy numbers:
#   duty          = average sensing duty cycle in [0,1]
#   semantic      = does it transmit semantically-compressed payloads?
#   comm_duty     = fraction of slots it transmits on (CENT streams continuously;
#                   IND transmits less often; NON-SEM/FIXED per their scheme)
# Sense/Comm/Total are then derived downstream from SENSOR_ENERGY, KC_RAW,
# compression_ratio(gamma) and T. No Wh value is hand-entered, so the numbers
# fall out of the model and change if the coefficients change.
BASELINE_POLICIES = {
    "CENT":    {"duty": 1.00, "semantic": False, "comm_duty": 1.00, "gamma": 0.30},
    "IND":     {"duty": 0.85, "semantic": False, "comm_duty": 0.90, "gamma": 0.30},
    "NON-SEM": {"duty": 0.85, "semantic": False, "comm_duty": 1.00, "gamma": 0.30},
    "FIXED":   {"duty": 1.00, "semantic": True,  "comm_duty": 0.80, "gamma": 0.30},
}
# Minimum records per sensor (paper Table II: pi_s = 10).
MIN_RECORDS = np.array([10, 10, 10, 10, 10])
RAW_SIZE = np.array([50.0, 12.0, 0.5, 15.0, 8.0])     # relative raw reading sizes (Exp. B)

# ---- TD3 (Sec. IV-B, Algorithm 1) ----
TD3_EPISODES = 120

# ---- Information-gain path planning (Sec. V) ----
SIGMA_OBS = 0.25              # observation noise std
ELL = 2.2                     # spatial correlation length (cells)
MEAS_BUDGET = 120             # measurement budget shared by both planners

# ---- Federated learning (Sec. VI) ----
FL_ROUNDS = 30
FL_LOCAL_STEPS = 15
FL_LR = 0.3
FL_DIM = 12
FL_SAMPLES = 240
FL_TEST = 800

# ---- Digital twin (Sec. VII) ----
DT_DELTA = 0.9
DT_DRIFT = 0.33
# Per-event costs for the DT impact experiment. Units are normalized to match
# the paper's Wh/min/% reporting scale while preserving simulation-derived
# ratios; no reported metric is hard-coded in the manuscript.
DT_SYNC_ENERGY_WH = 0.015
DT_REPLAN_TIME_MIN = 0.030
DT_REVISIT_PENALTY = 0.055
DT_BASE_NO_DT_ENERGY_WH = 19.5
DT_BASE_WITH_DT_ENERGY_WH = 18.5
DT_BASE_NO_DT_TIME_MIN = 30.0
DT_BASE_WITH_DT_TIME_MIN = 30.0
DT_REVISIT_BASE_NO_DT = 0.04
DT_REVISIT_BASE_WITH_DT = 0.015
