"""Cournot competition with linear inverse demand P = a - b*Q and constant marginal costs.

Gibbons, Game Theory for Applied Economists, ch. 1 (Cournot duopoly) and the n-firm
generalisation that is standard in an IO course. Firms choose quantities simultaneously;
each firm i's profit is (a - b*Q - c_i) * q_i where Q is total industry output.

Closed form. From firm i's first-order condition a - b*Q_{-i} - 2*b*q_i - c_i = 0, where
Q_{-i} = Q - q_i, rearranging gives b*(q_i + Q) = a - c_i, i.e. q_i = (a - c_i)/b - Q.
Summing over the n firms and solving for Q gives

    Q* = sum((a - c_i)/b) / (n + 1)
    q_i* = (a - c_i)/b - Q*

For symmetric costs this collapses to the textbook q_i = (a - c)/((n + 1)*b).
"""

import numpy as np


def _check_params(a, b, costs):
    if b <= 0:
        raise ValueError(f"slope b must be positive, got {b}")
    costs = np.atleast_1d(np.asarray(costs, dtype=float))
    if np.any(costs < 0):
        raise ValueError(f"marginal costs must be non-negative, got {costs}")
    if np.any(costs >= a):
        raise ValueError(f"marginal cost must be below the demand intercept a={a}, got {costs}")
    return costs


def price(total_quantity, a, b):
    return a - b * total_quantity


def profit(q_i, others_quantity, a, b, c_i):
    return (price(q_i + others_quantity, a, b) - c_i) * q_i


def best_response(others_quantity, a, b, c_i):
    """Firm i's profit-maximising output given the total output of its rivals."""
    _check_params(a, b, [c_i])
    return max(0.0, (a - c_i - b * others_quantity) / (2 * b))


def nash_quantities(a, b, costs):
    """Closed-form Cournot-Nash output for n firms with (possibly asymmetric) costs.

    Returns the vector of equilibrium quantities q_i. See module docstring for the
    derivation: Q* = sum((a - c_i)/b) / (n + 1), q_i* = (a - c_i)/b - Q*.
    """
    costs = _check_params(a, b, costs)
    n = len(costs)
    total = np.sum((a - costs) / b) / (n + 1)
    q = (a - costs) / b - total
    if np.any(q < 0):
        raise ValueError("computed a negative equilibrium quantity; costs too asymmetric for these parameters")
    return q


def symmetric_nash_quantity(n, a, b, c):
    """The textbook q_i = (a - c)/((n + 1)*b) for n identical firms."""
    if n < 1:
        raise ValueError(f"number of firms must be at least 1, got {n}")
    q = nash_quantities(a, b, [c] * n)
    return q[0]


def monopoly_quantity(a, b, c):
    """Total output a single firm (or a perfect cartel) would choose: (a - c)/(2b)."""
    _check_params(a, b, [c])
    return (a - c) / (2 * b)


def collusive_quantities(n, a, b, c):
    """Symmetric split of the monopoly output among n colluding firms."""
    if n < 1:
        raise ValueError(f"number of firms must be at least 1, got {n}")
    return np.full(n, monopoly_quantity(a, b, c) / n)


def competitive_quantity(a, b, c):
    """Perfectly competitive total output, where price equals marginal cost: (a - c)/b."""
    _check_params(a, b, [c])
    return (a - c) / b


def deviation_best_response_to_collusion(n, a, b, c):
    """A single firm's best response if the other n-1 stick to the collusive split.

    Used by repeated.py for the grim-trigger sustainability threshold: this is the
    one-shot payoff from cheating on a cartel agreement.
    """
    if n < 2:
        raise ValueError(f"deviation requires at least 2 firms, got {n}")
    others_quantity = (n - 1) * monopoly_quantity(a, b, c) / n
    return best_response(others_quantity, a, b, c)


def stackelberg(a, b, c1, c2):
    """Leader (firm 1) and follower (firm 2) quantities in 2-firm Stackelberg competition.

    The follower's reaction function is q2 = (a - c2 - b*q1)/(2b) (same as Cournot best
    response). Substituting into the leader's profit and maximising over q1 gives
    q1 = (a + c2 - 2*c1)/(2b). For symmetric costs this is the monopoly quantity
    (a - c)/(2b), and the follower produces half as much.
    """
    _check_params(a, b, [c1, c2])
    q1 = (a + c2 - 2 * c1) / (2 * b)
    if q1 < 0:
        raise ValueError("leader's quantity came out negative; costs too asymmetric for these parameters")
    q2 = best_response(q1, a, b, c2)
    return q1, q2


def simultaneous_best_response_dynamics(n, a, b, c, q0=None, n_rounds=20):
    """Iterated best-response dynamics with simultaneous updating, symmetric firms.

    Each round every firm best-responds to the *previous* round's rivals' total output.
    For symmetric firms this reduces to the scalar recursion
        q(t+1) = (a - c)/(2b) - ((n-1)/2) * q(t)
    which is a contraction (and converges) only when |(n-1)/2| < 1, i.e. n <= 2. At
    n = 3 the coefficient is exactly 1, giving sustained (non-decaying) oscillation
    rather than convergence, and for n >= 4 it diverges. This is the well-known failure
    of naive simultaneous best-response dynamics in Cournot oligopoly (see Gibbons ch. 1
    for the n=2 case; the n>=3 instability is a standard IO course exercise).

    Returns an array of shape (n_rounds + 1, n) with the quantity trajectory.
    """
    _check_params(a, b, [c])
    if n < 1:
        raise ValueError(f"number of firms must be at least 1, got {n}")
    if q0 is None:
        q0 = np.full(n, (a - c) / (2 * b) / n)
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    trajectory = np.zeros((n_rounds + 1, n))
    trajectory[0] = q0
    for t in range(n_rounds):
        q_prev = trajectory[t]
        total_prev = q_prev.sum()
        trajectory[t + 1] = [best_response(total_prev - q_prev[i], a, b, c) for i in range(n)]
    return trajectory
