"""Tests for bayesian.py: the Gibbons 3.1 Cournot-with-incomplete-information example."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import bayesian
import cournot

A, C1, CH, CL, THETA = 100.0, 20.0, 32.0, 20.0, 0.5


def test_closed_form_satisfies_the_three_best_response_conditions():
    q1, q2H, q2L = bayesian.closed_form(A, C1, CH, CL, THETA)
    expected_q2 = THETA * q2H + (1 - THETA) * q2L
    assert q1 == pytest.approx((A - C1 - expected_q2) / 2)
    assert q2H == pytest.approx((A - CH - q1) / 2)
    assert q2L == pytest.approx((A - CL - q1) / 2)


def test_numeric_solver_agrees_with_the_closed_form():
    closed = bayesian.closed_form(A, C1, CH, CL, THETA)
    numeric = bayesian.solve_numerically(A, C1, CH, CL, THETA)
    for a, b in zip(closed, numeric):
        assert a == pytest.approx(b, abs=1e-6)


def test_theta_zero_collapses_to_complete_information_with_the_low_cost():
    q1, q2H, q2L = bayesian.closed_form(A, C1, CH, CL, theta=0.0)
    expected = cournot.nash_quantities(A, 1.0, [C1, CL])
    assert q1 == pytest.approx(expected[0])
    assert q2L == pytest.approx(expected[1])


def test_theta_one_collapses_to_complete_information_with_the_high_cost():
    q1, q2H, q2L = bayesian.closed_form(A, C1, CH, CL, theta=1.0)
    expected = cournot.nash_quantities(A, 1.0, [C1, CH])
    assert q1 == pytest.approx(expected[0])
    assert q2H == pytest.approx(expected[1])


def test_high_cost_type_produces_less_than_low_cost_type():
    _, q2H, q2L = bayesian.closed_form(A, C1, CH, CL, THETA)
    assert q2H < q2L


def test_invalid_bayesian_parameters_raise():
    with pytest.raises(ValueError):
        bayesian.closed_form(A, C1, CH, CL, theta=1.5)
    with pytest.raises(ValueError):
        bayesian.closed_form(A, C1, cH=CL, cL=CH, theta=THETA)  # swapped high/low
