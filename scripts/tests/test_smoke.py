"""
Quick checks that the core functions behave.

    python -m pytest scripts/tests
or  python scripts/tests/test_smoke.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # config.py
sys.path.insert(0, str(HERE.parent))        # ibdlib.py

import numpy as np

import config as C
import ibdlib as L


def test_bh_matches_reference():

    p_values = np.array([
        0.01, 0.04, 0.03, 0.2, np.nan
    ])

    adjusted_values = C.bh(p_values)

    expected = [0.04, 0.0533333, 0.0533333, 0.2]

    assert np.allclose(
        adjusted_values[:4],
        expected,
        atol=1e-6
    )

    assert np.isnan(adjusted_values[4])


def test_cr1_matches_bruteforce():

    rng = np.random.default_rng(1)

    n = 300
    groups = 30

    # Each participant has 10 observations
    participant_ids = np.repeat(
        np.arange(groups),
        10
    )

    X = np.column_stack([
        np.ones(n),
        rng.normal(size=(n, 3))
    ])

    y = (
        X @ [1, 0.5, 0, -0.3]
        + rng.normal(size=n)
        + np.repeat(rng.normal(size=groups), 10)
    )

    # Calculate coefficients using the project function
    B, se, p, _ = L.ols_cr1(
        X,
        y[:, None],
        participant_ids
    )

    # Calculate the same values manually for comparison
    XtX_inverse = np.linalg.inv(X.T @ X)

    coefficients = np.linalg.solve(
        X.T @ X,
        X.T @ y
    )

    residuals = y - X @ coefficients

    meat = sum(
        np.outer(
            X[participant_ids == i].T @ residuals[participant_ids == i],
            X[participant_ids == i].T @ residuals[participant_ids == i]
        )
        for i in range(groups)
    )

    variance = (
        groups / (groups - 1)
        * (n - 1) / (n - 4)
        * XtX_inverse @ meat @ XtX_inverse
    )

    expected_se = np.sqrt(np.diag(variance))

    assert np.allclose(B[:, 0], coefficients)

    assert np.allclose(
        se[:, 0],
        expected_se
    )


def test_clr_rows_sum_to_zero():

    import pandas as pd

    rng = np.random.default_rng(0)

    # Generate random abundance values
    abundance = pd.DataFrame(
        rng.dirichlet(np.ones(6), 20)
        * (
            np.random.default_rng(1).random((20, 6)) > 0.4
        )
    )

    # Remove samples with zero total abundance
    abundance = abundance[
        abundance.sum(axis=1) > 0
    ]

    clr_values = L.clr(abundance)

    # CLR-transformed values should sum to zero for each sample
    assert np.allclose(
        clr_values.sum(axis=1),
        0
    )


def test_config_is_consistent():
    assert C.check_config() == []


def test_ffq_scoring_uses_config(monkeypatch=None):
    import pandas as pd

    answers = pd.Series(list(C.FFQ_SCORES) + [None], name="demo")
    scored = L.score_answers(answers)

    assert list(scored.iloc[:-1]) == list(C.FFQ_SCORES.values())
    assert scored.iloc[-1] != scored.iloc[-1]       # blank stays NaN


def test_unknown_ffq_answer_is_reported():
    import pandas as pd

    try:
        L.score_answers(pd.Series(["not a real answer"], name="demo"))
    except ValueError as err:
        assert "not a real answer" in str(err)
    else:
        raise AssertionError("unknown answer should raise")


def test_food_column_matching():
    cols = ["Water", "Water-based pick/jet", "Fruit juice (orange)", "Fruits (no juice) (Apples)"]

    assert L.find_food_column(cols, "Water")[0] == "Water"            # exact wins
    assert L.find_food_column(cols, "Fruit juice")[0] == "Fruit juice (orange)"
    assert L.find_food_column(cols, "Fruit")[0] is None                # ambiguous
    assert L.find_food_column(cols, "Beans")[0] is None                # missing


if __name__ == "__main__":

    tests = [
        test_bh_matches_reference,
        test_cr1_matches_bruteforce,
        test_clr_rows_sum_to_zero,
        test_config_is_consistent,
        test_ffq_scoring_uses_config,
        test_unknown_ffq_answer_is_reported,
        test_food_column_matching,
    ]

    for test in tests:
        test()
        print("ok", test.__name__)
