"""Tests for repeated.py: discounted payoffs and the grim-trigger threshold."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import cournot
import repeated


def test_finite_discounted_payoff_matches_the_geometric_series_closed_form():
    p, delta, n = 10.0, 0.9, 5
    direct = sum(delta ** t * p for t in range(n))
    assert repeated.finite_discounted_payoff(p, delta, n) == pytest.approx(direct)


def test_finite_discounted_payoff_with_no_discounting_is_just_the_sum():
    assert repeated.finite_discounted_payoff(10.0, 1.0, 5) == pytest.approx(50.0)


def test_infinite_discounted_payoff_is_the_limit_of_the_finite_sum():
    p, delta = 10.0, 0.9
    finite_large_n = repeated.finite_discounted_payoff(p, delta, 10_000)
    assert repeated.infinite_discounted_payoff(p, delta) == pytest.approx(finite_large_n, rel=1e-6)


def test_prisoners_dilemma_grim_trigger_threshold_on_a_standard_example():
    # T=5, R=3, P=1 -> delta* = (5-3)/(5-1) = 0.5
    delta_star = repeated.grim_trigger_threshold_prisoners_dilemma(R=3, T=5, P=1)
    assert delta_star == pytest.approx(0.5)


def test_prisoners_dilemma_threshold_requires_the_right_payoff_ordering():
    with pytest.raises(ValueError):
        repeated.grim_trigger_threshold_prisoners_dilemma(R=5, T=3, P=1)


def test_cournot_grim_trigger_threshold_for_duopoly_is_nine_seventeenths():
    delta_star = repeated.grim_trigger_threshold_cournot(2, 100.0, 1.0, 20.0)
    assert delta_star == pytest.approx(9 / 17)


def test_cournot_grim_trigger_threshold_matches_its_closed_form_for_several_n():
    for n in range(2, 8):
        direct = repeated.grim_trigger_threshold_cournot(n, 100.0, 1.0, 20.0)
        closed = repeated.grim_trigger_threshold_cournot_closed_form(n)
        assert direct == pytest.approx(closed)


def test_cournot_grim_trigger_threshold_is_independent_of_demand_and_cost_scale():
    # a, b, c only enter through (a-c)^2/b, which cancels out of the delta* ratio.
    t1 = repeated.grim_trigger_threshold_cournot(3, 100.0, 1.0, 20.0)
    t2 = repeated.grim_trigger_threshold_cournot(3, 50.0, 2.0, 10.0)
    assert t1 == pytest.approx(t2)


def test_deviation_profit_exceeds_collusive_profit_which_exceeds_nash_profit():
    # the ordering that makes the grim-trigger threshold well-defined in (0, 1)
    n, a, b, c = 2, 100.0, 1.0, 20.0
    collusive_q = cournot.monopoly_quantity(a, b, c) / n
    pi_collusive = cournot.profit(collusive_q, (n - 1) * collusive_q, a, b, c)
    dev_q = cournot.deviation_best_response_to_collusion(n, a, b, c)
    pi_dev = cournot.profit(dev_q, (n - 1) * collusive_q, a, b, c)
    nash_q = cournot.symmetric_nash_quantity(n, a, b, c)
    pi_nash = cournot.profit(nash_q, (n - 1) * nash_q, a, b, c)
    assert pi_dev > pi_collusive > pi_nash


def test_invalid_discount_factors_raise():
    with pytest.raises(ValueError):
        repeated.finite_discounted_payoff(10.0, 1.5, 5)
    with pytest.raises(ValueError):
        repeated.infinite_discounted_payoff(10.0, 1.0)
