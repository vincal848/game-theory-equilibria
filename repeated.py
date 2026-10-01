"""Discounted payoffs and grim-trigger sustainability in infinitely repeated games.

Gibbons, Game Theory for Applied Economists, ch. 2.3 (repeated games) and the folk
theorem discussion of grim-trigger strategies.
"""

import cournot


def finite_discounted_payoff(payoff, delta, n_periods):
    """Present value of a constant per-period payoff over n_periods, discounted at delta.

    Closed form of the finite geometric series: payoff * (1 - delta**n) / (1 - delta).
    """
    if not 0 <= delta <= 1:
        raise ValueError(f"discount factor must be in [0, 1], got {delta}")
    if n_periods < 0:
        raise ValueError(f"number of periods must be non-negative, got {n_periods}")
    if delta == 1:
        return payoff * n_periods
    return payoff * (1 - delta ** n_periods) / (1 - delta)


def infinite_discounted_payoff(payoff, delta):
    """Present value of a constant per-period payoff forever: payoff / (1 - delta)."""
    if not 0 <= delta < 1:
        raise ValueError(f"discount factor must be in [0, 1) for an infinite sum, got {delta}")
    return payoff / (1 - delta)


def grim_trigger_threshold_prisoners_dilemma(R, T, P):
    """Minimum discount factor sustaining mutual cooperation under grim trigger.

    R = payoff from mutual cooperation, T = temptation payoff from defecting against a
    cooperator, P = punishment payoff from mutual defection. Cooperating forever is at
    least as good as defecting once and being punished forever iff
        R / (1 - delta) >= T + delta * P / (1 - delta)
    which rearranges to delta >= (T - R) / (T - P). Requires T > R > P (otherwise
    defection is not actually tempting, or punishment is not actually worse).
    """
    if not T > R > P:
        raise ValueError(f"not a prisoner's dilemma payoff ordering: need T > R > P, got T={T}, R={R}, P={P}")
    return (T - R) / (T - P)


def grim_trigger_threshold_cournot(n, a, b, c):
    """Minimum discount factor sustaining the collusive (monopoly-split) output among
    n symmetric Cournot firms under grim trigger, reverting to the static Cournot-Nash
    quantity forever after a deviation.

    Derived the same way as the Prisoner's Dilemma threshold, using the three relevant
    per-firm profits: collusive, one-shot optimal deviation, and static Cournot-Nash
    punishment. delta* = (pi_dev - pi_collusive) / (pi_dev - pi_nash).
    """
    collusive_q = cournot.monopoly_quantity(a, b, c) / n
    pi_collusive = cournot.profit(collusive_q, (n - 1) * collusive_q, a, b, c)

    dev_q = cournot.deviation_best_response_to_collusion(n, a, b, c)
    pi_dev = cournot.profit(dev_q, (n - 1) * collusive_q, a, b, c)

    nash_q = cournot.symmetric_nash_quantity(n, a, b, c)
    pi_nash = cournot.profit(nash_q, (n - 1) * nash_q, a, b, c)

    return (pi_dev - pi_collusive) / (pi_dev - pi_nash)


def grim_trigger_threshold_cournot_closed_form(n):
    """Closed form delta* = (n+1)^2 / (n^2 + 6n + 1), independent of a, b, c.

    This is an algebraic simplification of grim_trigger_threshold_cournot: all three
    profits scale with (a-c)^2/b, so that factor cancels out of the ratio. At n = 2,
    delta* = 9/17. Kept separate from the profit-based version so tests can check they
    agree, rather than trusting the simplification on its own.
    """
    if n < 2:
        raise ValueError(f"collusion requires at least 2 firms, got {n}")
    return (n + 1) ** 2 / (n ** 2 + 6 * n + 1)
