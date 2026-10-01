"""Tests for cournot.py: closed-form Nash, benchmarks, Stackelberg, dynamics."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

import cournot

A, B, C = 100.0, 1.0, 20.0


def test_best_response_matches_the_original_example():
    # The legacy module's own worked example: firm 2 produces 30, firm 1's best
    # response is (100 - 20 - 30) / 2 = 25.
    assert cournot.best_response(30, A, B, C) == pytest.approx(25.0)


def test_symmetric_nash_quantity_matches_the_closed_form_for_n_1_to_5():
    for n in range(1, 6):
        expected = (A - C) / ((n + 1) * B)
        assert cournot.symmetric_nash_quantity(n, A, B, C) == pytest.approx(expected)


def test_n_equals_1_is_the_monopoly_quantity():
    monopoly = cournot.monopoly_quantity(A, B, C)
    assert cournot.symmetric_nash_quantity(1, A, B, C) == pytest.approx(monopoly)


def test_total_quantity_approaches_the_competitive_level_as_n_grows():
    competitive = cournot.competitive_quantity(A, B, C)
    totals = [n * cournot.symmetric_nash_quantity(n, A, B, C) for n in [1, 5, 20, 200, 2000]]
    # monotonically increasing towards, and converging on, the competitive total
    assert all(t2 > t1 for t1, t2 in zip(totals, totals[1:]))
    assert totals[-1] == pytest.approx(competitive, rel=1e-2)
    assert totals[-1] < competitive


def test_nash_quantities_handles_asymmetric_costs():
    q = cournot.nash_quantities(A, B, [10.0, 30.0])
    # firm with the lower cost produces more
    assert q[0] > q[1]
    # both firms' first-order conditions hold at this quantity vector
    total = q.sum()
    for i, c_i in enumerate([10.0, 30.0]):
        others = total - q[i]
        assert q[i] == pytest.approx(cournot.best_response(others, A, B, c_i))


def test_collusive_output_is_half_the_competitive_output():
    # Textbook result for linear demand and constant marginal cost: monopoly quantity
    # is half the competitive quantity, independent of how it's split among firms.
    collusive_total = cournot.collusive_quantities(3, A, B, C).sum()
    assert collusive_total == pytest.approx(cournot.competitive_quantity(A, B, C) / 2)


def test_stackelberg_leader_produces_the_monopoly_quantity_under_symmetric_costs():
    q1, q2 = cournot.stackelberg(A, B, C, C)
    assert q1 == pytest.approx(cournot.monopoly_quantity(A, B, C))
    assert q2 == pytest.approx(cournot.monopoly_quantity(A, B, C) / 2)
    # the leader produces strictly more than under simultaneous Cournot
    assert q1 > cournot.symmetric_nash_quantity(2, A, B, C)


def test_best_response_dynamics_converges_for_two_firms():
    traj = cournot.simultaneous_best_response_dynamics(2, A, B, C, q0=[5.0, 5.0], n_rounds=40)
    nash_q = cournot.symmetric_nash_quantity(2, A, B, C)
    assert traj[-1, 0] == pytest.approx(nash_q, abs=1e-6)
    assert traj[-1, 1] == pytest.approx(nash_q, abs=1e-6)


def test_best_response_dynamics_oscillates_without_decay_for_three_firms():
    traj = cournot.simultaneous_best_response_dynamics(3, A, B, C, q0=[5.0, 5.0, 5.0], n_rounds=40)
    nash_q = cournot.symmetric_nash_quantity(3, A, B, C)
    # it does not converge: it keeps swinging around the Nash quantity with an
    # amplitude that does not shrink between the last and second-to-last cycle
    late_amplitude = abs(traj[-1, 0] - nash_q)
    earlier_amplitude = abs(traj[-3, 0] - nash_q)
    assert late_amplitude == pytest.approx(earlier_amplitude, abs=1e-6)
    assert late_amplitude > 5.0


def test_best_response_dynamics_diverges_for_four_firms():
    traj = cournot.simultaneous_best_response_dynamics(4, A, B, C, q0=[15.0, 15.0, 15.0, 15.0], n_rounds=10)
    # the swings grow round over round (before hitting the q >= 0 floor)
    early_swing = abs(traj[2, 0] - traj[1, 0])
    later_swing = abs(traj[6, 0] - traj[5, 0])
    assert later_swing > early_swing


def test_invalid_cournot_parameters_raise():
    with pytest.raises(ValueError):
        cournot.best_response(10, A, b=0, c_i=C)
    with pytest.raises(ValueError):
        cournot.symmetric_nash_quantity(0, A, B, C)
    with pytest.raises(ValueError):
        cournot.best_response(10, A, B, c_i=-5)
    with pytest.raises(ValueError):
        cournot.best_response(10, A, B, c_i=A + 1)
