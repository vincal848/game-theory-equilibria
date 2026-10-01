"""Tests for normal_form.py: pure/mixed Nash equilibria, zero-sum value, dominance."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

import normal_form as nf

MATCHING_PENNIES = (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]]))
BATTLE_OF_SEXES = (np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]]))
PRISONERS_DILEMMA = (np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]]))
ROCK_PAPER_SCISSORS = (
    np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
    np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
)


def test_matching_pennies_mixed_equilibrium_is_one_half_one_half():
    A, B = MATCHING_PENNIES
    assert nf.pure_nash_equilibria(A, B) == []
    equilibria = nf.mixed_nash_equilibria(A, B)
    assert len(equilibria) == 1
    p, q = equilibria[0]
    assert p == pytest.approx([0.5, 0.5])
    assert q == pytest.approx([0.5, 0.5])


def test_battle_of_the_sexes_has_three_equilibria():
    A, B = BATTLE_OF_SEXES
    pure = nf.pure_nash_equilibria(A, B)
    assert set(pure) == {(0, 0), (1, 1)}
    equilibria = nf.mixed_nash_equilibria(A, B)
    assert len(equilibria) == 3
    strictly_mixed = [(p, q) for p, q in equilibria if 0 < p[0] < 1]
    assert len(strictly_mixed) == 1
    p, q = strictly_mixed[0]
    # textbook values for this payoff matrix: p = (2/3, 1/3), q = (1/3, 2/3)
    assert p == pytest.approx([2 / 3, 1 / 3])
    assert q == pytest.approx([1 / 3, 2 / 3])


def test_prisoners_dilemma_has_a_unique_equilibrium_at_mutual_defection():
    A, B = PRISONERS_DILEMMA
    assert nf.pure_nash_equilibria(A, B) == [(1, 1)]
    equilibria = nf.mixed_nash_equilibria(A, B)
    assert len(equilibria) == 1
    p, q = equilibria[0]
    assert p == pytest.approx([0, 1])
    assert q == pytest.approx([0, 1])


def test_rock_paper_scissors_equilibrium_is_uniform():
    A, B = ROCK_PAPER_SCISSORS
    assert nf.pure_nash_equilibria(A, B) == []
    equilibria = nf.mixed_nash_equilibria(A, B)
    assert len(equilibria) == 1
    p, q = equilibria[0]
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert q == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_zero_sum_value_of_matching_pennies_is_zero():
    A, _ = MATCHING_PENNIES
    value, p = nf.zero_sum_value(A)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx([0.5, 0.5])


def test_zero_sum_value_matches_a_known_worked_example():
    # Standard 2x2 zero-sum example with no saddle point: value = (a11*a22 - a12*a21)
    # / (a11 + a22 - a12 - a21) = (4*3 - 2*1) / (4 + 3 - 2 - 1) = 2.5
    A = np.array([[4, 2], [1, 3]])
    value, _ = nf.zero_sum_value(A)
    assert value == pytest.approx(2.5)


def test_zero_sum_value_handles_non_positive_payoffs():
    # This is exactly what the legacy LP got wrong: it was unbounded for any matrix
    # containing a non-positive entry, including this one.
    A = np.array([[-2, 3], [1, -1]])
    value, p = nf.zero_sum_value(A)
    assert np.isfinite(value)
    assert p.sum() == pytest.approx(1.0)


def test_dominance_elimination_reduces_a_dominated_game():
    # Row 1 (payoffs [0, 0]) is strictly dominated by row 0 ([1, 1]) for player 1.
    # Player 2 is indifferent between columns, so only the row is eliminated.
    A = np.array([[1, 1], [0, 0]])
    B = np.array([[1, 1], [1, 1]])
    rows, cols = nf.dominated_strategies_eliminated(A, B)
    assert rows == [0]
    assert cols == [0, 1]


def test_mismatched_payoff_matrix_shapes_raise():
    with pytest.raises(ValueError):
        nf.pure_nash_equilibria(np.array([[1, 2]]), np.array([[1, 2], [3, 4]]))
