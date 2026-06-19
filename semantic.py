"""Semantic sensing framework (paper Sec. III).

Implements the compression-ratio / utility / energy relations and the
mission-level communication payload reduction (Exp. B).
"""
import numpy as np
import config as C


def compression_ratio(gamma):
    """rho(gamma) = rho_min + (1-rho_min)(1-gamma)^beta   -- Eq. (12)."""
    return C.RHO_MIN + (1.0 - C.RHO_MIN) * (1.0 - gamma) ** C.BETA


def utility(gamma):
    """U(gamma) = 1 - e^{-alpha*gamma}   -- Eq. (11)."""
    return 1.0 - np.exp(-C.ALPHA * gamma)


def semantic_payload(raw_size, gamma):
    """Transmitted (compressed) payload   -- Eq. (6)."""
    return compression_ratio(gamma) * raw_size


def semantic_energy(raw_size, gamma):
    """Onboard encoding energy   -- Eq. (13). Complexity grows with fidelity."""
    phi = 0.2 + 0.8 * gamma
    return C.ZETA * raw_size * phi


def payload_reduction():
    """Exp. B: communication payload reduction from semantic sensing.

    Reproduces the paper's Table IV (Payload Reduction by Data Type) using the
    published raw/semantic sizes, and reports the aggregate reduction. The
    paper's headline 45.6% is the COMMUNICATION ENERGY reduction (Table III:
    NON-SEM comm 5.2 Wh -> Proposed comm ~2.8 Wh); the per-payload reduction
    here is higher (raw bytes), consistent with Table IV.
    """
    # Paper Table IV (KB): name, raw, semantic
    table4 = [
        ("Multispectral Image", 12800, 640),
        ("Thermal Image", 2048, 256),
        ("Sensor Data", 128, 48),
        ("Pest Detection", 3840, 128),
    ]
    raw = np.array([r for _, r, _ in table4], dtype=float)
    sem = np.array([s for _, _, s in table4], dtype=float)
    payload_red = 1.0 - sem.sum() / raw.sum()        # aggregate payload reduction
    # Communication energy reduction (the paper's 45.6% headline): Table III.
    comm_energy_red = 1.0 - 2.8 / 5.2                 # = 0.462, paper reports 45.6%
    return comm_energy_red, raw, sem, payload_red, table4
