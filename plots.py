"""Figures for docs/img/. Kept out of the pricing/solving path -- nothing here runs on
import, and nothing in cournot.py, normal_form.py, bayesian.py or repeated.py imports
matplotlib. Run directly to regenerate the figures:

    python plots.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cournot

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "img")

A, B, C = 100.0, 1.0, 20.0


def best_response_curves():
    """The hero figure: both firms' best-response curves in Cournot duopoly, crossing
    exactly at the Cournot-Nash equilibrium.
    """
    q = np.linspace(0, 50, 200)
    br1 = [cournot.best_response(q2, A, B, C) for q2 in q]  # firm 1's response to q2
    br2 = [cournot.best_response(q1, A, B, C) for q1 in q]  # firm 2's response to q1

    nash_q = cournot.symmetric_nash_quantity(2, A, B, C)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q, br1, label=r"Firm 1 best response: $q_1(q_2)$")
    ax.plot(br2, q, label=r"Firm 2 best response: $q_2(q_1)$")
    ax.plot(nash_q, nash_q, "ko", markersize=8, zorder=5,
            label=f"Nash equilibrium ({nash_q:.2f}, {nash_q:.2f})")
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.set_title("Cournot duopoly best-response curves (a=100, b=1, c=20)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "best_response_curves.png"), dpi=150)
    plt.close(fig)


def best_response_dynamics():
    """Simultaneous-update best-response dynamics: converges for n=2, sustained
    oscillation at n=3, diverging (then clipped at zero) for n=4.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, n in zip(axes, [2, 3, 4]):
        traj = cournot.simultaneous_best_response_dynamics(n, A, B, C, q0=[5.0] * n, n_rounds=20)
        nash_q = cournot.symmetric_nash_quantity(n, A, B, C)
        ax.plot(traj[:, 0], marker="o", markersize=3)
        ax.axhline(nash_q, color="gray", linestyle="--", label=f"Nash q={nash_q:.2f}")
        ax.set_title(f"n = {n} firms")
        ax.set_xlabel("round")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("firm 1's quantity")
    fig.suptitle("Simultaneous best-response dynamics: stable only for n <= 2")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "best_response_dynamics.png"), dpi=150)
    plt.close(fig)


def profit_surface_3d():
    """Optional: firm 1's profit as a function of both quantities, same parameters as
    the legacy plot_cournot_3d (replaces brf_3dplot.png).
    """
    q1 = np.linspace(0, 50, 60)
    q2 = np.linspace(0, 50, 60)
    Q1, Q2 = np.meshgrid(q1, q2)
    profit1 = (cournot.price(Q1 + Q2, A, B) - C) * Q1

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Q1, Q2, profit1, cmap="viridis")
    ax.set_xlabel("Firm 1 quantity")
    ax.set_ylabel("Firm 2 quantity")
    ax.set_zlabel("Firm 1 profit")
    ax.set_title("Firm 1 profit surface (a=100, b=1, c=20)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "profit_surface_3d.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    best_response_curves()
    best_response_dynamics()
    profit_surface_3d()
    print(f"Wrote figures to {IMG_DIR}")
