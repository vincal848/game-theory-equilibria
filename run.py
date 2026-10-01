"""Command-line entry point.

    python run.py cournot -a 100 -b 1 -c 20 -n 3
    python run.py nash --game battle-of-the-sexes
    python run.py bayes -a 100 --c1 20 --cH 32 --cL 20 --theta 0.5
    python run.py repeated --pd --T 5 --R 3 --P 1
    python run.py repeated --cournot -n 2 -a 100 -b 1 -c 20
"""

import argparse

import numpy as np

import bayesian
import cournot
import normal_form
import repeated

GAMES = {
    "matching-pennies": (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])),
    "battle-of-the-sexes": (np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]])),
    "prisoners-dilemma": (np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]])),
    "rock-paper-scissors": (
        np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
        np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
    ),
}


def cmd_cournot(args):
    q = cournot.nash_quantities(args.a, args.b, [args.c] * args.n)
    total = q.sum()
    p = cournot.price(total, args.a, args.b)
    collusive = cournot.collusive_quantities(args.n, args.a, args.b, args.c)
    competitive = cournot.competitive_quantity(args.a, args.b, args.c)

    print(f"Cournot, n={args.n} firms, a={args.a}, b={args.b}, c={args.c}")
    print(f"  Nash quantity per firm: {q[0]:.4f}  (total Q = {total:.4f}, price = {p:.4f})")
    print(f"  Collusive quantity per firm: {collusive[0]:.4f}  (total Q = {collusive.sum():.4f})")
    print(f"  Competitive total Q: {competitive:.4f}")

    if args.n == 2:
        q1, q2 = cournot.stackelberg(args.a, args.b, args.c, args.c)
        print(f"  Stackelberg (n=2): leader = {q1:.4f}, follower = {q2:.4f}")


def cmd_nash(args):
    A, B = GAMES[args.game]
    print(f"Game: {args.game}")
    print("Pure Nash equilibria (row, col):", normal_form.pure_nash_equilibria(A, B))
    for p, q in normal_form.mixed_nash_equilibria(A, B):
        print(f"  mixed equilibrium: p={np.round(p, 4)}  q={np.round(q, 4)}")
    if np.array_equal(B, -A):
        value, p = normal_form.zero_sum_value(A)
        print(f"  zero-sum value: {value:.4f}, optimal row strategy {np.round(p, 4)}")


def cmd_bayes(args):
    q1, q2H, q2L = bayesian.closed_form(args.a, args.c1, args.cH, args.cL, args.theta)
    print(f"Bayesian Cournot (Gibbons 3.1): a={args.a}, c1={args.c1}, cH={args.cH}, "
          f"cL={args.cL}, theta={args.theta}")
    print(f"  q1 = {q1:.4f}")
    print(f"  q2(high cost) = {q2H:.4f}")
    print(f"  q2(low cost)  = {q2L:.4f}")


def cmd_repeated(args):
    if args.pd:
        delta = repeated.grim_trigger_threshold_prisoners_dilemma(args.R, args.T, args.P)
        print(f"Prisoner's dilemma grim-trigger threshold: delta* = {delta:.4f}")
    if args.cournot:
        delta = repeated.grim_trigger_threshold_cournot(args.n, args.a, args.b, args.c)
        closed = repeated.grim_trigger_threshold_cournot_closed_form(args.n)
        print(f"Cournot (n={args.n}) grim-trigger threshold: delta* = {delta:.4f} "
              f"(closed form {closed:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cournot = sub.add_parser("cournot")
    p_cournot.add_argument("-a", type=float, default=100.0)
    p_cournot.add_argument("-b", type=float, default=1.0)
    p_cournot.add_argument("-c", type=float, default=20.0)
    p_cournot.add_argument("-n", type=int, default=2)
    p_cournot.set_defaults(func=cmd_cournot)

    p_nash = sub.add_parser("nash")
    p_nash.add_argument("--game", choices=sorted(GAMES), default="battle-of-the-sexes")
    p_nash.set_defaults(func=cmd_nash)

    p_bayes = sub.add_parser("bayes")
    p_bayes.add_argument("-a", type=float, default=100.0)
    p_bayes.add_argument("--c1", type=float, default=20.0)
    p_bayes.add_argument("--cH", type=float, default=32.0)
    p_bayes.add_argument("--cL", type=float, default=20.0)
    p_bayes.add_argument("--theta", type=float, default=0.5)
    p_bayes.set_defaults(func=cmd_bayes)

    p_repeated = sub.add_parser("repeated")
    p_repeated.add_argument("--pd", action="store_true", help="prisoner's dilemma threshold")
    p_repeated.add_argument("--T", type=float, default=5.0)
    p_repeated.add_argument("--R", type=float, default=3.0)
    p_repeated.add_argument("--P", type=float, default=1.0)
    p_repeated.add_argument("--cournot", action="store_true", help="Cournot collusion threshold")
    p_repeated.add_argument("-n", type=int, default=2)
    p_repeated.add_argument("-a", type=float, default=100.0)
    p_repeated.add_argument("-b", type=float, default=1.0)
    p_repeated.add_argument("-c", type=float, default=20.0)
    p_repeated.set_defaults(func=cmd_repeated)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
