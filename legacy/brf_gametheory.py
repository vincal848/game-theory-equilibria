"""SUPERSEDED. Kept for reference only -- this file is not part of the package and
is known to be incorrect.

Coursework implementation motivated by Gibbons, Game Theory for Applied Economists,
ch. 1, 3, 4, and an IO course. Do not import it.

Known defects, each with a named regression test:

1. ``multi_firm_cournot`` minimises the *negative of total industry profit*, i.e. it
   maximises total industry profit, which is the collusive (cartel) output, not the
   Cournot-Nash equilibrium where each firm maximises its own profit taking rivals'
   quantities as given. For n=3, a=100, c=20, b=1 it returns q_i = 13.33 (total
   Q = 40, the monopoly split) where the Cournot-Nash answer is q_i = 20 (total
   Q = 60). See ``tests/test_cournot.py::test_symmetric_nash_quantity_matches_the_closed_form_for_n_1_to_5``
   and ``docs/VALIDATION.md``.
2. ``nash_equilibrium`` sets the LP objective to ``c = [-1, ..., -1]``, i.e. it
   *maximises* sum(x) subject only to lower-bound constraints, which is unbounded for
   essentially any input -- including the module's own worked example,
   ``nash_equilibrium([[3, 1], [0, 2]])``, which returns ``None``. The correct LP
   minimises sum(x) subject to ``A^T x >= 1``, and even then is only the right question
   to ask for a zero-sum game, which this one isn't. See
   ``tests/test_normal_form.py::test_zero_sum_value_matches_a_known_worked_example``.
3. ``mixed_strategy_nash`` computes ``player_payoffs - opponent_payoffs`` and feeds
   that into the same broken LP. Subtracting the two payoff matrices is only a
   meaningful reduction to a zero-sum game when the game actually is zero-sum
   (``opponent_payoffs == -player_payoffs``); for a general-sum game such as Battle of
   the Sexes it answers a different game than the one that was asked about, which
   ``normal_form.mixed_nash_equilibria`` (support enumeration, correct for nondegenerate
   games) replaces. See
   ``tests/test_normal_form.py::test_battle_of_the_sexes_has_three_equilibria``.
4. ``bayesian_nash_equilibrium`` never uses ``opponent_type`` inside
   ``type_specific_utility`` for the *strategy* argument -- it calls
   ``self.calculate_best_response(opponent_strategy=0)`` unconditionally, so every
   type's best response is computed against a rival producing zero, regardless of the
   type probabilities or the rival's actual type-contingent strategy. ``bayesian.py``
   replaces it with the Gibbons 3.1 closed form and a fixed-point solver that agree to
   6 decimal places. See
   ``tests/test_bayesian.py::test_closed_form_satisfies_the_three_best_response_conditions``.
5. ``num_firms`` and ``repeated`` are constructor attributes that are never read
   anywhere in the class. Module-level example code runs on import, including two
   ``plt.show()`` calls, so simply importing this file blocks on a plot window.

The working implementation is the flat modules at the repository root: ``cournot.py``,
``normal_form.py``, ``bayesian.py``, ``repeated.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

class BestResponseCalculator:
    def __init__(self, player_utility, strategy_space, num_firms=2, repeated=False):
        self.player_utility = player_utility
        self.strategy_space = strategy_space
        self.num_firms = num_firms
        self.repeated = repeated

    def calculate_best_response(self, opponent_strategy):
        best_strategy = None
        max_utility = -np.inf

        
        for strategy in self.strategy_space:
            utility = self.player_utility(strategy, opponent_strategy)
            if utility > max_utility:
                max_utility = utility
                best_strategy = strategy

        return best_strategy

    def cournot_best_response(self, opponent_quantity, price_intercept=100, cost=20):
        def utility(q1):
            price = price_intercept - (q1 + opponent_quantity)
            profit = (price - cost) * q1
            return profit

        self.player_utility = utility
        return self.calculate_best_response(opponent_quantity)

    def multi_firm_cournot(self, num_firms, cost=20, price_intercept=100):
        def total_profit(q):
            total = 0
            for i in range(num_firms):
                q_i = q[i]
                q_others = np.delete(q, i)
                total_quantity = q_i + sum(q_others)
                price = price_intercept - total_quantity
                profit = (price - cost) * q_i
                total -= profit  
            return total

        initial_guess = [10] * num_firms
        bounds = [(0, 50) for _ in range(num_firms)]
        result = minimize(total_profit, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            return result.x
        else:
            return None

    def nash_equilibrium(self, utility_matrix):
        num_strategies = len(utility_matrix)
        c = [-1] * num_strategies
        A_ub = -np.transpose(utility_matrix)
        b_ub = [-1] * len(utility_matrix[0])
        bounds = [(0, None) for _ in range(num_strategies)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            return res.x / sum(res.x)
        else:
            return None

    def mixed_strategy_nash(self, player_payoffs, opponent_payoffs):
        combined_payoffs = player_payoffs - opponent_payoffs
        
        num_strategies = len(player_payoffs)
        c = [-1] * num_strategies
        A_ub = -combined_payoffs
        b_ub = [-1] * len(combined_payoffs[0])
        bounds = [(0, None) for _ in range(num_strategies)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            return res.x / sum(res.x)
        else:
            return None

    def bayesian_nash_equilibrium(self, player_types, opponent_types, utility_function):
      
        bayesian_nash = {}

        for player_type in player_types:
            best_responses = []
            for opponent_type in opponent_types:
                def type_specific_utility(player_strategy, opponent_strategy):
                    return utility_function(player_type, opponent_type, player_strategy, opponent_strategy)

                self.player_utility = type_specific_utility
                best_response = self.calculate_best_response(opponent_strategy=0) 
                best_responses.append(best_response)

            bayesian_nash[player_type] = np.mean(best_responses) 

        return bayesian_nash

    def discounted_payoff(self, payoff_function, strategy, discount_factor=0.9, num_periods=100):
        total_payoff = 0
        for t in range(num_periods):
            discounted_value = (discount_factor ** t) * payoff_function(strategy)
            total_payoff += discounted_value
        return total_payoff

    def plot_best_response(self, opponent_quantity, price_intercept=100, cost=20):
        best_responses = []
        opponent_quantities = np.linspace(0, 50, 100)
        for q2 in opponent_quantities:
            best_responses.append(self.cournot_best_response(q2, price_intercept, cost))

        plt.plot(opponent_quantities, best_responses, label='Best Response Curve')
        plt.xlabel('Opponent Quantity')
        plt.ylabel('Best Response Quantity')
        plt.title('Best Response Curve in Cournot Competition')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cournot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        q1 = np.linspace(0, 50, 50)
        q2 = np.linspace(0, 50, 50)
        q1, q2 = np.meshgrid(q1, q2)
        price = 100 - q1 - q2
        profit1 = (price - 20) * q1

        ax.plot_surface(q1, q2, profit1, cmap='viridis')
        ax.set_xlabel('Firm 1 Quantity')
        ax.set_ylabel('Firm 2 Quantity')
        ax.set_zlabel('Profit for Firm 1')
        plt.title('3D Surface Plot for Firm 1 Profit')
        plt.show()

# Example:

strategy_space = np.linspace(0, 50, 500)


calculator = BestResponseCalculator(player_utility=None, strategy_space=strategy_space, num_firms=2, repeated=False)

opponent_quantity = 30


best_response_quantity = calculator.cournot_best_response(opponent_quantity)

print(f"The best response for Firm 1, given that Firm 2 produces {opponent_quantity} units, is to produce {best_response_quantity:.2f} units.")


calculator.plot_best_response(opponent_quantity)

# multi-firm cournot
num_firms = 3
equilibrium_quantities = calculator.multi_firm_cournot(num_firms)
print(f"Equilibrium quantities for {num_firms} firms: {equilibrium_quantities}")


calculator.plot_cournot_3d()

# Nash Equilibrium Example
utility_matrix = [[3, 1], [0, 2]]
nash_strategy = calculator.nash_equilibrium(utility_matrix)
print(f"The mixed strategy Nash equilibrium for Player 1 is: {nash_strategy}")

# Examply Bayesian Nash
def bayesian_utility(player_type, opponent_type, player_strategy, opponent_strategy):
    return (100 - player_strategy - opponent_strategy) * player_strategy - 20 * player_strategy

player_types = [1, 2]
opponent_types = [1, 2]

bayesian_nash = calculator.bayesian_nash_equilibrium(player_types, opponent_types, bayesian_utility)
print(f"The Bayesian Nash equilibrium strategies are: {bayesian_nash}")
