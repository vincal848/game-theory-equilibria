"""Finite two-player normal-form (bimatrix) games.

Gibbons ch. 1 (pure-strategy Nash equilibrium, dominance) and ch. 3's treatment of
mixed strategies. Player 1's payoff matrix A and player 2's payoff matrix B are both
shaped (m, n): A[i, j] and B[i, j] are the payoffs when player 1 plays row i and
player 2 plays column j. Zero-sum games are the special case B = -A.
"""

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


def _check_game(A: ArrayLike, B: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"payoff matrices must have the same shape, got {A.shape} and {B.shape}")
    if A.ndim != 2:
        raise ValueError("payoff matrices must be two-dimensional")
    return A, B


def pure_nash_equilibria(A: ArrayLike, B: ArrayLike) -> list[tuple[int, int]]:
    """All pure-strategy Nash equilibria, found by marking best responses.

    Cell (i, j) is an equilibrium if row i is a best response to column j (for
    player 1) and column j is a best response to row i (for player 2).
    """
    A, B = _check_game(A, B)
    m, n = A.shape
    equilibria = []
    for i in range(m):
        for j in range(n):
            row_best = A[i, j] >= A[:, j].max() - 1e-12
            col_best = B[i, j] >= B[i, :].max() - 1e-12
            if row_best and col_best:
                equilibria.append((i, j))
    return equilibria


def dominated_strategies_eliminated(A: ArrayLike, B: ArrayLike) -> tuple[list[int], list[int]]:
    """Iterated elimination of strictly dominated pure strategies.

    Returns (rows, cols): the row and column indices of the original game that
    survive. A row i is strictly dominated if some other surviving row i' gives
    player 1 a strictly higher payoff against every surviving column, and symmetrically
    for columns and player 2.
    """
    A, B = _check_game(A, B)
    rows = list(range(A.shape[0]))
    cols = list(range(A.shape[1]))
    changed = True
    while changed:
        changed = False
        for i in list(rows):
            others = [r for r in rows if r != i]
            if any(np.all(A[np.ix_([r], cols)] > A[np.ix_([i], cols)]) for r in others):
                rows.remove(i)
                changed = True
        for j in list(cols):
            others = [c for c in cols if c != j]
            if any(np.all(B[np.ix_(rows, [c])] > B[np.ix_(rows, [j])]) for c in others):
                cols.remove(j)
                changed = True
    return rows, cols


def zero_sum_value(A: ArrayLike) -> tuple[float, NDArray[np.float64]]:
    """Value and optimal row-player strategy of a zero-sum game via linear programming.

    Player 1 (rows) is the maximiser, player 2 (columns) the minimiser, and
    player 2's payoff matrix is -A. The standard LP (Dantzig, see also Gibbons ch. 1's
    discussion of maxmin strategies) requires a strictly positive payoff matrix, so
    we shift by a constant M before solving and subtract it from the value afterwards.
    This shift is what the legacy code was missing; without it the LP is unbounded for
    any matrix containing a non-positive entry (see legacy/brf_gametheory.py).

    Returns (value, p) where p is player 1's optimal mixed strategy.
    """
    A = np.asarray(A, dtype=float)
    shift = 1.0 - A.min()  # smallest shifted entry is 1, strictly positive
    shifted = A + shift
    m, n = shifted.shape
    # minimize sum(x) subject to shifted^T @ x >= 1 (elementwise), x >= 0
    c = np.ones(m)
    A_ub = -shifted.T
    b_ub = -np.ones(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * m, method="highs")
    if not res.success:
        raise ValueError(f"LP did not solve: {res.message}")
    x = res.x
    shifted_value = 1.0 / x.sum()
    p = x * shifted_value
    return shifted_value - shift, p


def _indifference_solve(
    payoffs: NDArray[np.float64], support_self: list[int], support_other: list[int]
) -> tuple[NDArray[np.float64], float] | None:
    """Solve for the mixed strategy over support_other that makes support_self indifferent.

    payoffs is the payoff matrix of the player being made indifferent (rows = that
    player's strategies, columns = the opponent's). Returns the probability vector over
    support_other (summing to 1) and the resulting equalised payoff, or None if the
    linear system is singular.
    """
    k = len(support_other)
    sub = payoffs[np.ix_(support_self, support_other)]
    if sub.shape[0] != k:
        return None  # support enumeration requires equal-size supports (nondegenerate game)
    # k-1 indifference equations (row i vs row 0) + 1 normalisation equation
    rows = []
    rhs = []
    for r in range(1, k):
        rows.append(sub[r] - sub[0])
        rhs.append(0.0)
    rows.append(np.ones(k))
    rhs.append(1.0)
    M = np.array(rows)
    try:
        probs = np.linalg.solve(M, np.array(rhs))
    except np.linalg.LinAlgError:
        return None
    if np.any(probs < -1e-9):
        return None
    probs = np.clip(probs, 0, None)
    value = sub[0] @ probs
    return probs, value


def mixed_nash_equilibria(
    A: ArrayLike, B: ArrayLike, tol: float = 1e-9
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """All mixed-strategy Nash equilibria of a nondegenerate bimatrix game, by support
    enumeration (Gibbons ch. 3's indifference principle, done exhaustively rather than
    guessing a support).

    For every pair of equal-size supports (one for each player), solve for the mixed
    strategy over each support that makes the opponent indifferent between the support
    strategies, then check that (a) the probabilities are non-negative and (b) no
    strategy outside the support does strictly better. This is exact for nondegenerate
    games (distinct payoffs, no unnecessary ties) and includes pure equilibria as the
    singleton-support case.

    Returns a list of (p, q) pairs, p over player 1's rows and q over player 2's columns.
    """
    A, B = _check_game(A, B)
    m, n = A.shape
    equilibria = []
    for k in range(1, min(m, n) + 1):
        for support1 in itertools.combinations(range(m), k):
            for support2 in itertools.combinations(range(n), k):
                q_result = _indifference_solve(A, list(support1), list(support2))
                if q_result is None:
                    continue
                q_support, v1 = q_result
                p_result = _indifference_solve(B.T, list(support2), list(support1))
                if p_result is None:
                    continue
                p_support, v2 = p_result

                p = np.zeros(m)
                p[list(support1)] = p_support
                q = np.zeros(n)
                q[list(support2)] = q_support

                # No row outside the support may beat the equalised payoff v1.
                off_support_rows = [i for i in range(m) if i not in support1]
                if any(A[i] @ q > v1 + tol for i in off_support_rows):
                    continue
                off_support_cols = [j for j in range(n) if j not in support2]
                if any(p @ B[:, j] > v2 + tol for j in off_support_cols):
                    continue

                candidate = (tuple(np.round(p, 9)), tuple(np.round(q, 9)))
                if candidate not in [(tuple(np.round(pe, 9)), tuple(np.round(qe, 9))) for pe, qe in equilibria]:
                    equilibria.append((p, q))
    return equilibria
