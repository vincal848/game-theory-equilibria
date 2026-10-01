"""Regenerate docs/VALIDATION.md and the figures in docs/img/.

Every number in the documentation comes from this script, so the docs cannot drift
away from the code:

    python validate.py

Run from the repository root.
"""

import io
from pathlib import Path

import numpy as np

import bayesian
import cournot
import normal_form as nf
import plots
import repeated

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

A, B, C = 100.0, 1.0, 20.0


def section_cournot(out: io.StringIO) -> None:
    out.write("## Cournot-Nash quantities, a=100, b=1, c=20\n\n")
    out.write("| n firms | q_i (Nash) | Total Q | Price | Collusive q_i | Collusive Q | Competitive Q |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    for n in [1, 2, 3, 5, 10, 50]:
        q = cournot.symmetric_nash_quantity(n, A, B, C)
        total = n * q
        price = cournot.price(total, A, B)
        collusive = cournot.collusive_quantities(n, A, B, C)
        competitive = cournot.competitive_quantity(A, B, C)
        out.write(f"| {n} | {q:.4f} | {total:.4f} | {price:.4f} | {collusive[0]:.4f} "
                  f"| {collusive.sum():.4f} | {competitive:.4f} |\n")
    out.write("\n")

    out.write("### Legacy multi_firm_cournot vs. the correct closed form (n=3)\n\n")
    from scipy.optimize import minimize

    def total_profit(q, num_firms=3, cost=C, price_intercept=A):
        total = 0
        for i in range(num_firms):
            q_i = q[i]
            q_others = np.delete(q, i)
            total_quantity = q_i + sum(q_others)
            price = price_intercept - total_quantity
            profit = (price - cost) * q_i
            total -= profit
        return total

    legacy_result = minimize(total_profit, [10] * 3, bounds=[(0, 50)] * 3, method="SLSQP")
    correct = cournot.symmetric_nash_quantity(3, A, B, C)
    out.write(f"Legacy `multi_firm_cournot(3)` returns `{np.round(legacy_result.x, 4)}` "
              f"(total Q = {legacy_result.x.sum():.4f}) -- the collusive output, because it "
              f"maximises total industry profit rather than each firm's own profit.\n")
    out.write(f"Correct Cournot-Nash: q_i = {correct:.4f} each, total Q = {3 * correct:.4f}.\n\n")

    out.write("### Stackelberg vs. simultaneous Cournot (n=2)\n\n")
    q1, q2 = cournot.stackelberg(A, B, C, C)
    nash2 = cournot.symmetric_nash_quantity(2, A, B, C)
    out.write(f"Stackelberg leader = {q1:.4f}, follower = {q2:.4f}, total = {q1 + q2:.4f}. "
              f"Simultaneous Cournot (both firms) = {nash2:.4f} each, total = {2 * nash2:.4f}.\n\n")


def section_dynamics(out: io.StringIO) -> None:
    out.write("## Simultaneous best-response dynamics\n\n")
    out.write("| n firms | round 18 | round 19 | round 20 | behaviour |\n")
    out.write("|---|---|---|---|---|\n")
    for n, label in [(2, "converges"), (3, "sustained oscillation"), (4, "diverges, clipped at 0")]:
        traj = cournot.simultaneous_best_response_dynamics(n, A, B, C, q0=[5.0] * n, n_rounds=20)
        out.write(f"| {n} | {traj[18, 0]:.4f} | {traj[19, 0]:.4f} | {traj[20, 0]:.4f} | {label} |\n")
    out.write("\n")


def section_normal_form(out: io.StringIO) -> None:
    out.write("## Normal-form games\n\n")
    games = {
        "matching pennies": (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])),
        "battle of the sexes": (np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]])),
        "prisoner's dilemma": (np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]])),
        "rock-paper-scissors": (
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
            np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
        ),
    }
    out.write("| Game | Pure NE | Mixed NE |\n")
    out.write("|---|---|---|\n")
    for name, (A_, B_) in games.items():
        pure = nf.pure_nash_equilibria(A_, B_)
        mixed = nf.mixed_nash_equilibria(A_, B_)
        mixed_str = "; ".join(f"p={np.round(p, 3).tolist()}, q={np.round(q, 3).tolist()}" for p, q in mixed)
        out.write(f"| {name} | {pure} | {mixed_str} |\n")
    out.write("\n")

    out.write("### Legacy nash_equilibrium / mixed_strategy_nash vs. correct zero_sum_value\n\n")
    out.write("The legacy LP's objective had the wrong sign and no normalising constraint, "
              "so it is unbounded for every matrix tried, including its own worked example:\n\n")
    from scipy.optimize import linprog

    def legacy_nash_equilibrium(utility_matrix):
        num_strategies = len(utility_matrix)
        c = [-1] * num_strategies
        A_ub = -np.transpose(utility_matrix)
        b_ub = [-1] * len(utility_matrix[0])
        bounds = [(0, None) for _ in range(num_strategies)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return res.x / sum(res.x) if res.success else None

    legacy_matrix = [[3, 1], [0, 2]]
    legacy_result = legacy_nash_equilibrium(legacy_matrix)
    out.write(f"`legacy_nash_equilibrium({legacy_matrix})` -> `{legacy_result}` (unbounded LP).\n\n")

    value, p = nf.zero_sum_value(np.array(legacy_matrix))
    out.write(f"`zero_sum_value({legacy_matrix})` (treating it as zero-sum) -> "
              f"value={value:.4f}, p={np.round(p, 4).tolist()}.\n\n")

    value_known, _ = nf.zero_sum_value(np.array([[4, 2], [1, 3]]))
    out.write(f"Known worked example `[[4, 2], [1, 3]]`: value = {value_known:.4f} "
              f"(formula value = (4*3-2*1)/(4+3-2-1) = 2.5).\n\n")


def section_bayesian(out: io.StringIO) -> None:
    out.write("## Bayesian Cournot (Gibbons 3.1), a=100, c1=20, cH=32, cL=20\n\n")
    out.write("| theta | q1 | q2(high) | q2(low) |\n")
    out.write("|---|---|---|---|\n")
    for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q1, q2H, q2L = bayesian.closed_form(A, 20.0, 32.0, 20.0, theta)
        out.write(f"| {theta} | {q1:.4f} | {q2H:.4f} | {q2L:.4f} |\n")
    out.write("\n")


def section_repeated(out: io.StringIO) -> None:
    out.write("## Grim-trigger thresholds\n\n")
    out.write("| n firms | delta* (from profits) | delta* (closed form) |\n")
    out.write("|---|---|---|\n")
    for n in range(2, 8):
        direct = repeated.grim_trigger_threshold_cournot(n, A, B, C)
        closed = repeated.grim_trigger_threshold_cournot_closed_form(n)
        out.write(f"| {n} | {direct:.4f} | {closed:.4f} |\n")
    out.write("\n")
    pd_delta = repeated.grim_trigger_threshold_prisoners_dilemma(R=3, T=5, P=1)
    out.write(f"Prisoner's dilemma (T=5, R=3, P=1): delta* = {pd_delta:.4f}.\n\n")


def main() -> None:
    out = io.StringIO()
    out.write("# Validation\n\n")
    out.write("Generated by `validate.py`. Do not edit by hand.\n\n")
    section_cournot(out)
    section_dynamics(out)
    section_normal_form(out)
    section_bayesian(out)
    section_repeated(out)

    (DOCS / "VALIDATION.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"Wrote {DOCS / 'VALIDATION.md'}")

    import os
    os.makedirs(plots.IMG_DIR, exist_ok=True)
    plots.best_response_curves()
    plots.best_response_dynamics()
    plots.profit_surface_3d()
    print(f"Wrote figures to {plots.IMG_DIR}")


if __name__ == "__main__":
    main()
