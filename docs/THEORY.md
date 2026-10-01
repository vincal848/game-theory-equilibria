# Theory

Derivations behind each module. References are to Gibbons, *Game Theory for Applied
Economists* (1992), by chapter and section, plus one standard IO course result for the
collusion threshold.

## Cournot competition (`cournot.py`, Gibbons ch. 1.1)

Linear inverse demand `P = a - b*Q`, firm i has constant marginal cost `c_i`. Firm i's
profit is `(a - b*(q_i + Q_{-i}) - c_i) * q_i`, where `Q_{-i}` is the total output of
every other firm. The first-order condition is

```
a - b*Q_{-i} - 2*b*q_i - c_i = 0
```

Since `Q_{-i} = Q - q_i`, this rearranges to `b*(q_i + Q) = a - c_i`, i.e.

```
q_i = (a - c_i)/b - Q
```

Summing over the n firms and solving for `Q`:

```
Q* = sum_i[(a - c_i)/b] / (n + 1)
q_i* = (a - c_i)/b - Q*
```

For identical firms (`c_i = c`), this is the textbook `q_i = (a - c)/((n + 1)*b)`.

**Monopoly / collusion.** A single firm (or a cartel splitting output) maximises total
profit `(a - b*Q - c)*Q`, giving `Q = (a - c)/(2*b)` -- exactly half the competitive
output below.

**Perfect competition.** Price equals marginal cost, `a - b*Q = c`, so
`Q = (a - c)/b`. As `n -> infinity`, the Cournot total `n*(a-c)/((n+1)*b)` converges to
this from below.

**Stackelberg (ch. 2.1).** The leader (firm 1) picks `q1` anticipating the follower's
best response `q2 = (a - c2 - b*q1)/(2b)`. Substituting into firm 1's profit and
maximising over `q1` gives `q1 = (a + c2 - 2*c1)/(2b)`; for symmetric costs this is the
monopoly quantity, and the follower produces half of that.

**Best-response dynamics.** With symmetric firms and simultaneous updating, firm i's
quantity next round is its best response to the *current* total of the other n-1
firms, each sitting at the same `q(t)`:

```
q(t+1) = (a - c)/(2b) - ((n-1)/2) * q(t)
```

This is an affine recursion `x(t+1) = A - B*x(t)` with `B = (n-1)/2`. It converges iff
`|B| < 1`, i.e. `n < 3`. At `n = 3`, `B = 1` exactly, giving sustained (non-decaying)
oscillation; for `n >= 4`, `B > 1` and the amplitude grows each round (until the
non-negativity constraint clips it). Verified numerically in
`tests/test_cournot.py` and plotted in `docs/img/best_response_dynamics.png`.

## Normal-form games (`normal_form.py`, Gibbons ch. 1.1, 3.1)

**Pure Nash equilibria** are found by marking, for each cell, whether it is a best
response for both players simultaneously -- the direct implementation of Gibbons'
definition.

**Mixed Nash equilibria** are found by support enumeration (Gibbons ch. 3.1's
indifference principle, applied exhaustively rather than guessing a support). For a
candidate pair of equal-size supports, the opponent's indifference condition pins down
a probability vector via a linear system (the indifference equations plus
normalisation). The result is accepted only if both probability vectors are
non-negative and no strategy outside its support does strictly better. This is exact
for nondegenerate games.

**Zero-sum value.** With player 2's payoffs equal to `-A`, the maxmin value and
optimal row strategy solve a linear program. The standard formulation (Dantzig)
requires a strictly positive payoff matrix:

```
minimize sum(x)  subject to  A^T x >= 1 (elementwise),  x >= 0
```

then `value = 1/sum(x)`, `p = x * value`. Since real payoff matrices are not generally
positive, `zero_sum_value` shifts the matrix by `M = 1 - min(A)` before solving and
subtracts `M` back out of the value afterwards. This shift, plus using `minimize`
rather than `maximize` for the objective, is exactly what the legacy LP was missing
(see `legacy/brf_gametheory.py`).

## Bayesian Cournot (`bayesian.py`, Gibbons ch. 3.1 "An Example")

Linear demand `P = a - Q`. Firm 1's cost `c1` is common knowledge. Firm 2's cost is
`cH` with probability `theta` and `cL` with probability `1 - theta`, known to firm 2
but not firm 1. Firm 1 picks one quantity `q1` maximising *expected* profit:

```
q1 = (a - c1 - E[q2]) / 2,   E[q2] = theta*q2H + (1-theta)*q2L
```

Each type of firm 2 best-responds to `q1` exactly as under complete information:

```
q2H = (a - cH - q1)/2,   q2L = (a - cL - q1)/2
```

Substituting `E[q2]` into firm 1's condition and solving the three equations together:

```
q1 = (a - 2*c1 + theta*cH + (1-theta)*cL) / 3
```

At `theta = 0` or `theta = 1` this collapses exactly to the complete-information
Cournot-Nash quantities with `c2 = cL` or `c2 = cH` respectively, which is how
`tests/test_bayesian.py` cross-checks it against `cournot.nash_quantities`.

## Repeated games (`repeated.py`, Gibbons ch. 2.3)

**Discounted payoffs.** A constant per-period payoff `p` discounted at `delta` over
`n` periods is the finite geometric series `p*(1 - delta^n)/(1 - delta)`; over an
infinite horizon it is `p/(1 - delta)`.

**Grim-trigger threshold, Prisoner's Dilemma.** Cooperating forever beats defecting
once (getting the temptation payoff `T`) and being punished forever (getting `P` every
period after) iff

```
R/(1-delta) >= T + delta*P/(1-delta)
=> delta >= (T - R)/(T - P)
```

requiring the usual ordering `T > R > P`.

**Grim-trigger threshold, symmetric Cournot collusion.** Same argument with three
per-firm profits: collusive (monopoly split), one-shot optimal deviation against
rivals still playing the collusive quantity, and static Cournot-Nash punishment
forever after:

```
delta* = (pi_dev - pi_collusive) / (pi_dev - pi_nash)
```

Working through the algebra (the deviator's best response to `Q_{-i} = (n-1)*q_C`,
where `q_C = (a-c)/(2bn)` is the collusive quantity) gives deviation quantity
`q_dev = (a-c)*(n+1)/(4*b*n)` and, using `price - c = b*q` at any best response,
`pi_dev = b*q_dev^2`. All three profits scale with `(a-c)^2/b`, which cancels out of
the ratio, leaving the closed form

```
delta* = (n+1)^2 / (n^2 + 6n + 1)
```

At `n = 2` this is `9/17 ~= 0.5294`. `repeated.py` keeps the profit-based computation
(`grim_trigger_threshold_cournot`) and the closed form
(`grim_trigger_threshold_cournot_closed_form`) as two independent routes, and the test
suite checks they agree, rather than trusting the algebraic simplification on its own.
