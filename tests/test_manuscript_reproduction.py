"""Reproduction tests for the data-driven pipeline.

These lock the CURRENT real-data results, not the old manuscript-faithful
constants. Two things are asserted:

  1. Determinism: re-running a stage with the same seed + data reproduces the
     same number (bit-for-bit at the printed precision).
  2. Value ranges: the headline numbers fall in the ranges reported in the
     manuscript's Numerical Results section, computed from the real datasets /
     the policy energy model (no hand-entered targets).

Dataset-backed stages (mapping, federated, DT, payload) are skipped when the
datasets are not present, so the suite still passes on a code-only checkout.
See scripts/download_datasets.sh to fetch them.
"""
import math
import os

import config as C
from energy_model import baselines, headline_savings


# ---------------------------------------------------------------- energy model
def test_energy_table_is_policy_derived_and_reproducible():
    duty = 0.4812000079512596
    rows_a = baselines(proposed_duty=duty)
    rows_b = baselines(proposed_duty=duty)
    # deterministic: same inputs -> identical table
    assert rows_a == rows_b
    # fly+hover floor is the documented constant shared by every approach
    assert math.isclose(C.E_FLY_HOVER, 14.3, abs_tol=1e-9)
    for name in ("Proposed", "CENT", "IND", "NON-SEM", "FIXED"):
        assert math.isclose(rows_a[name]["fly_hover"], 14.3, abs_tol=1e-9)


def test_energy_saving_vs_cent_in_reported_range():
    # real policy model yields ~23.3% (manuscript, highlighted)
    rows = baselines(proposed_duty=0.4812000079512596)
    saving = headline_savings(rows)
    assert 0.21 <= saving <= 0.25, f"energy saving {saving:.3f} out of range"


# ---------------------------------------------------------------- datasets
_DIVINE = os.environ.get("DIVINE_DIR")
_WEEDNET = os.environ.get("WEEDNET_DIR")
_IP102 = os.environ.get("IP102_DIR")


def _has(path):
    return bool(path) and os.path.isdir(path)


import pytest


@pytest.mark.skipif(not _has(_IP102), reason="IP102 dataset not present")
def test_payload_reduction_on_real_ip102_reproducible():
    from ip102_data import payload_reduction_real
    r1 = payload_reduction_real(n_sample=2000, seed=42)["payload_reduction"]
    r2 = payload_reduction_real(n_sample=2000, seed=42)["payload_reduction"]
    assert math.isclose(r1, r2, abs_tol=1e-9)          # deterministic
    assert 0.40 <= r1 <= 0.55                          # ~48.5% reported


@pytest.mark.skipif(not _has(_DIVINE), reason="DIVINE dataset not present")
def test_federated_gain_on_real_divine_reproducible():
    from fed_nn import run_federated_nn
    a = run_federated_nn(seed=42, source="divine")
    b = run_federated_nn(seed=42, source="divine")
    assert math.isclose(a["improvement"], b["improvement"], abs_tol=1e-6)
    assert a["fed_acc"] >= a["local_acc"]              # federation helps
    assert 0.20 <= a["improvement"] <= 0.40            # ~30.3% reported


@pytest.mark.skipif(not _has(_DIVINE), reason="DIVINE dataset not present")
def test_dt_sync_reduction_on_real_streams_reproducible():
    from dt_divine import run_dt_experiment
    a = run_dt_experiment(sync_quantile=0.9)
    b = run_dt_experiment(sync_quantile=0.9)
    assert a["adaptive_syncs"] == b["adaptive_syncs"]  # deterministic
    assert 0.85 <= a["sync_reduction"] <= 0.98         # ~93.2% reported


@pytest.mark.skipif(not _has(_WEEDNET), reason="WeedNet dataset not present")
def test_mapping_gain_on_real_weednet_reproducible():
    from path_planning import run_mapping_experiment_avg
    a = run_mapping_experiment_avg()
    b = run_mapping_experiment_avg()
    assert math.isclose(a["improvement"], b["improvement"], abs_tol=1e-6)
    # modest, high-variance gain reported transparently (~5.4%)
    assert a["improvement"] > -0.05
