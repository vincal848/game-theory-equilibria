"""Cournot duopoly with incomplete information about costs.

Gibbons, Game Theory for Applied Economists, ch. 3.1 ("An Example"): linear demand
P = a - Q, firm 1's cost c1 is common knowledge, firm 2's cost is cH with probability
theta and cL with probability (1 - theta), known to firm 2 but not to firm 1. Firm 1
picks a single quantity q1; firm 2 picks a quantity contingent on its type, q2H or q2L.

Closed form. Firm 1 maximises expected profit treating E[q2] as given:
    q1 = (a - c1 - E[q2]) / 2
Each type of firm 2 best-responds to firm 1's q1 exactly as in the complete-information
case:
    q2H = (a - cH - q1) / 2,   q2L = (a - cL - q1) / 2
Substituting E[q2] = theta*q2H + (1-theta)*q2L into firm 1's condition and solving the
three equations simultaneously gives
    q1 = (a - 2*c1 + theta*cH + (1 - theta)*cL) / 3
"""

import numpy as np


def _check_params(a, c1, cH, cL, theta):
    if not 0 <= theta <= 1:
        raise ValueError(f"theta must be a probability in [0, 1], got {theta}")
    if cH < cL:
        raise ValueError(f"cH must be the high-cost type and cL the low-cost type, got cH={cH} < cL={cL}")
    if min(c1, cL) < 0 or max(c1, cH) >= a:
        raise ValueError("costs must be non-negative and below the demand intercept a")


def closed_form(a, c1, cH, cL, theta):
    """The Gibbons 3.1 closed-form equilibrium: (q1, q2H, q2L)."""
    _check_params(a, c1, cH, cL, theta)
    expected_c2 = theta * cH + (1 - theta) * cL
    q1 = (a - 2 * c1 + expected_c2) / 3
    q2H = (a - cH - q1) / 2
    q2L = (a - cL - q1) / 2
    return q1, q2H, q2L


def solve_numerically(a, c1, cH, cL, theta, tol=1e-12, max_iter=10_000):
    """Fixed-point iteration on the same three best-response conditions as closed_form,
    used as an independent cross-check that does not rely on having solved the algebra
    by hand correctly.
    """
    _check_params(a, c1, cH, cL, theta)
    q1, q2H, q2L = 0.0, 0.0, 0.0
    for _ in range(max_iter):
        expected_q2 = theta * q2H + (1 - theta) * q2L
        new_q1 = (a - c1 - expected_q2) / 2
        new_q2H = (a - cH - new_q1) / 2
        new_q2L = (a - cL - new_q1) / 2
        if max(abs(new_q1 - q1), abs(new_q2H - q2H), abs(new_q2L - q2L)) < tol:
            q1, q2H, q2L = new_q1, new_q2H, new_q2L
            break
        q1, q2H, q2L = new_q1, new_q2H, new_q2L
    return q1, q2H, q2L
