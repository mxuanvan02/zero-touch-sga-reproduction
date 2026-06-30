import math

import config as C
from energy_model import baselines, headline_savings
from semantic import payload_reduction
from scalability import scalability


def test_energy_table_matches_manuscript_convention():
    rows = baselines(proposed_duty=0.4812000079512596)

    assert math.isclose(C.E_FLY_HOVER, 14.3, abs_tol=1e-9)
    assert math.isclose(rows["Proposed"]["sense"], 3.42, abs_tol=0.02)
    assert math.isclose(rows["Proposed"]["comm"], 2.83, abs_tol=0.02)
    assert math.isclose(rows["Proposed"]["total"], 20.55, abs_tol=0.03)
    assert math.isclose(rows["CENT"]["total"], 25.10, abs_tol=0.01)
    assert math.isclose(headline_savings(rows), 0.181, abs_tol=0.002)


def test_semantic_payload_reduction_matches_manuscript_scale():
    reduction, *_ = payload_reduction()
    assert math.isclose(reduction, 0.462, abs_tol=0.002)


def test_scalability_energy_reduction_matches_manuscript_scale():
    _, reduction = scalability((1, 3, 5, 10))
    assert math.isclose(reduction, 0.355, abs_tol=0.002)
