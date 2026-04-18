"""
fig_E_multiplicative.py
=======================
Reproduces Figure E of ECSoC Paper 2.

Multiplicative noise mechanism — analytical predictions only.

Panel 1 : Additive vs multiplicative Kramers P_self predictions compared
          with empirical values (Suppress and Late-rise)
Panel 2 : Noise amplitude profile σ(ψ) = ε₀ exp(-κψ²)
Panel 3 : Effective Stratonovich potential V_eff(ψ) = V(ψ) + D*κ*ψ²
Panel 4 : Illustrative Heun trajectories under multiplicative noise
          (qualitative; NOT used for P_self estimation)

[All P_self values in Panel 1 are analytical Kramers predictions.
Direct simulation validation in the large-noise regime is designated for
Paper 3. See manuscript Section 6.2 and 6.3.]

Random seed: SEED = 42

Usage
-----
    python fig_E_multiplicative.py
Output : ./output/fig_E_multiplicative.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from langevin_core import (
    D, B, kappa, epsilon0, MARKOV_SELF_TRANS, STATE_A,
    potential, effective_potential_stratonovich,
    sigma_multiplicative,
    kramers_self_transition_prob,
    kramers_self_transition_multiplicative,
    heun_stratonovich,
    THR_SUP, THR_LR,
)

SEED = 42
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SEC = 16.0

# States to compare
STATES_COMPARE = ["Suppress", "Late-rise"]


def main():
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(15, 8))
    gs  = GridSpec(1, 4, figure=fig, wspace=0.38)

    psi_grid = np.linspace(-1.2, 1.2, 400)

    # ── Panel 1: P_self comparison ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    P_additive = {
        s: kramers_self_transition_prob(STATE_A[s], D, WINDOW_SEC)
        for s in STATES_COMPARE
    }
    P_multiplicative = {
        s: kramers_self_transition_multiplicative(STATE_A[s], D, WINDOW_SEC)
        for s in STATES_COMPARE
    }
    P_empirical = {s: MARKOV_SELF_TRANS[s] for s in STATES_COMPARE}

    x      = np.arange(len(STATES_COMPARE))
    w      = 0.22
    colors = {"Additive":      "#1E88E5",
              "Multiplicative": "#E53935",
              "Empirical":     "#757575"}

    ax1.bar(x - w,   [P_additive[s]      for s in STATES_COMPARE], w,
            label="Additive (Kramers)",      color=colors["Additive"],      alpha=0.85)
    ax1.bar(x,       [P_multiplicative[s] for s in STATES_COMPARE], w,
            label="Multiplicative (Kramers)", color=colors["Multiplicative"], alpha=0.85)
    ax1.bar(x + w,   [P_empirical[s]      for s in STATES_COMPARE], w,
            label="Empirical (Paper 1)",      color=colors["Empirical"],     alpha=0.75)

    # Annotate values
    for i, s in enumerate(STATES_COMPARE):
        ax1.text(i - w,   P_additive[s]      + 0.01, f"{P_additive[s]:.2f}",
                 ha="center", fontsize=6.5)
        ax1.text(i,       P_multiplicative[s] + 0.01, f"{P_multiplicative[s]:.2f}",
                 ha="center", fontsize=6.5)
        ax1.text(i + w,   P_empirical[s]      + 0.01, f"{P_empirical[s]:.2f}",
                 ha="center", fontsize=6.5)

    ax1.set_xticks(x); ax1.set_xticklabels(STATES_COMPARE, fontsize=8)
    ax1.set_ylabel("Self-transition probability", fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("P_self: Additive vs\nMultiplicative noise",
                  fontsize=8, fontweight="bold")
    ax1.legend(fontsize=6, loc="upper right")
    ax1.text(0.05, 0.05,
             "All P_self: analytical\nKramers predictions.\nSimulation validation\n→ Paper 3",
             transform=ax1.transAxes, fontsize=6, va="bottom",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    # ── Panel 2: noise amplitude profile σ(ψ) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sigma_vals = np.array([sigma_multiplicative(p) for p in psi_grid])
    ax2.plot(psi_grid, sigma_vals, "r-", lw=2.0, label=r"$\sigma(\psi)=\varepsilon_0 e^{-\kappa\psi^2}$")
    ax2.axhline(epsilon0, color="gray", lw=1.0, ls="--", alpha=0.7,
                label=f"ε₀={epsilon0:.3f}")

    # Mark fixed points for Suppress and Late-rise
    for s, color in [("Suppress", "#1E88E5"), ("Late-rise", "#E53935")]:
        A_s    = STATE_A[s]
        if A_s <= 0:
            continue
        psi_s  = np.sqrt(A_s / B)
        sig_s  = sigma_multiplicative(psi_s)
        ax2.plot([-psi_s, psi_s], [sig_s, sig_s], "o", color=color, ms=7,
                 zorder=5, label=f"ψ*({s})")
        ax2.annotate(f"ψ*={psi_s:.3f}\nσ={sig_s:.3f}\n(-{(1-sig_s/epsilon0)*100:.0f}%)",
                     xy=(psi_s, sig_s), fontsize=5.5,
                     xytext=(8, -12), textcoords="offset points",
                     color=color)

    ax2.set_xlabel("ψ", fontsize=8)
    ax2.set_ylabel("σ(ψ)", fontsize=8)
    ax2.set_title(f"Noise amplitude profile\n(κ={kappa}, ε₀={epsilon0:.3f})",
                  fontsize=8, fontweight="bold")
    ax2.legend(fontsize=6)

    # ── Panel 3: effective potential ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for s, color, A_val in [
        ("Suppress",   "#1E88E5", STATE_A["Suppress"]),
        ("Late-rise",  "#E53935", STATE_A["Late-rise"]),
    ]:
        if A_val <= 0:
            continue
        V_add  = potential(psi_grid, A_val)
        V_eff  = effective_potential_stratonovich(psi_grid, A_val, D)
        # Shift minimum to 0
        V_add -= V_add.min()
        V_eff -= V_eff.min()
        ax3.plot(psi_grid, V_add, "--", color=color, lw=1.4, alpha=0.6,
                 label=f"V  ({s})")
        ax3.plot(psi_grid, V_eff, "-",  color=color, lw=2.0,
                 label=f"V_eff ({s})")

    ax3.set_xlabel("ψ", fontsize=8)
    ax3.set_ylabel("Potential (shifted)", fontsize=8)
    ax3.set_title("V_eff = V + Dκψ²\n(multiplicative deepens wells)",
                  fontsize=8, fontweight="bold")
    ax3.legend(fontsize=6, loc="upper center")
    ax3.text(0.05, 0.05,
             "Deeper ΔV_eff →\nlonger Kramers τ →\nhigher P_self",
             transform=ax3.transAxes, fontsize=6, va="bottom",
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ── Panel 4: illustrative Heun trajectories ───────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])

    traj_params = [
        dict(A=STATE_A["Late-rise"], psi0=-0.35,
             color="#E53935", label="Late-rise (A=+0.15)"),
        dict(A=STATE_A["Suppress"],  psi0=-0.50,
             color="#1E88E5", label="Suppress (A=−0.20)"),
    ]
    n_traj = 2500
    dt_t   = 0.1

    for tp in traj_params:
        psi_t = heun_stratonovich(tp["psi0"], tp["A"], D,
                                  n_traj, dt_t, rng=rng)
        t_t   = np.arange(n_traj + 1) * dt_t
        ax4.plot(t_t, psi_t, color=tp["color"], lw=0.8, alpha=0.85,
                 label=tp["label"])

    ax4.axhline(THR_LR,  color="red",  lw=0.8, ls="--", alpha=0.6)
    ax4.axhline(THR_SUP, color="blue", lw=0.8, ls="--", alpha=0.6)
    ax4.axhline(0,       color="gray", lw=0.5, ls=":")
    ax4.set_xlabel("t (min)", fontsize=8)
    ax4.set_ylabel("ψ (CHI)", fontsize=8)
    ax4.set_title("Illustrative Heun trajectories\n(multiplicative noise)",
                  fontsize=8, fontweight="bold")
    ax4.legend(fontsize=6)
    ax4.text(0.05, 0.05,
             "Qualitative illustration;\nNOT used for P_self\nestimation.",
             transform=ax4.transAxes, fontsize=6, va="bottom",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    fig.suptitle(
        "Figure E  |  Multiplicative Noise Mechanism — Analytical Predictions Only\n"
        "[Panel 1: analytical Kramers predictions. Simulation validation → Paper 3. "
        "SEED=42]",
        fontsize=9
    )

    out_path = os.path.join(OUTPUT_DIR, "fig_E_multiplicative.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
