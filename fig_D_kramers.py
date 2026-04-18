"""
fig_D_kramers.py
================
Reproduces Figure D of ECSoC Paper 2.

Parameter estimation and Kramers escape-time validation.

Top-left    : Path 1 — chronic R² deterioration simulation
Top-centre  : A(R²) profile during chronic deterioration
Top-right   : Path 2 — noise-induced Suppress→Late-rise escape simulation
Bottom-left : Kramers escape time τ vs A for four noise levels D=0.04–0.12
Bottom-centre: Predicted vs observed self-transition probabilities
Bottom-right : Parameter estimation summary table

[All estimates exploratory. Kramers formula applied in large-noise regime
(ΔV/D ≈ 0.17–0.33); provides directional ordering, not quantitative
precision. See manuscript Sections 3.3, 6.2, 6.3.]

Random seed: SEED = 42

Usage
-----
    python fig_D_kramers.py
Output : ./output/fig_D_kramers.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from langevin_core import (
    D, Rc, a0, B, MARKOV_SELF_TRANS, STATE_A,
    simulate_path1_chronic, euler_maruyama,
    kramers_escape_time, kramers_self_transition_prob,
    barrier_height, control_A,
    THR_SUP, THR_LR
)

SEED = 42
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SEC = 16.0   # Markov window (seconds)


def main():
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Top-left: Path 1 simulation ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    sim = simulate_path1_chronic(rng=rng)
    psi = sim["psi"];  R2  = sim["R2"];  A   = sim["A"];  t = sim["t"]

    # Colour by R² zone
    zone_color = np.where(R2 > 0.96, "#43A047",
                 np.where(R2 > 0.93, "#FB8C00", "#E53935"))
    for i in range(len(psi) - 1):
        ax.plot(t[i:i+2], psi[i:i+2], color=zone_color[i], lw=0.8)

    ax.axhline(THR_LR,  color="red",  lw=0.8, ls="--", alpha=0.7)
    ax.axhline(THR_SUP, color="blue", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("t (min)", fontsize=8)
    ax.set_ylabel("ψ (CHI)", fontsize=8)
    ax.set_title("Path 1: Chronic R² deterioration", fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#43A047", label="R²>0.96"),
        Patch(color="#FB8C00", label="0.93<R²<0.96"),
        Patch(color="#E53935", label="R²<0.93"),
    ], fontsize=6, loc="upper right")

    # ── Top-centre: A(R²) profile ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, A, lw=1.6, color="#1565C0")
    ax.axhline(0, color="red", lw=1.0, ls="--", alpha=0.7, label="A=0 (bifurcation)")
    ax.fill_between(t, A, 0, where=(A > 0), alpha=0.20, color="green",
                    label="Bistable (A>0)")
    ax.fill_between(t, A, 0, where=(A < 0), alpha=0.20, color="red",
                    label="Monostable (A<0)")
    ax.set_xlabel("t (min)", fontsize=8)
    ax.set_ylabel("A(R²)", fontsize=8)
    ax.set_title("A(R²) during deterioration", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)

    # ── Top-right: Path 2 simulation ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    A2     = +0.22
    D2     = D
    n2     = 4000
    dt2    = 0.1
    psi2   = euler_maruyama(-0.40, A2, D2, n2, dt2, rng=rng)
    t2     = np.arange(n2 + 1) * dt2
    # Detect transition: first sustained crossing of THR_LR
    in_lr  = psi2 >= THR_LR
    trans_i = None
    for i in range(50, len(psi2)):
        if np.all(in_lr[i-20:i]):
            trans_i = i - 20
            break
    colors2 = np.where(
        np.arange(len(psi2)) < (trans_i if trans_i else len(psi2)),
        "#1E88E5", "#E53935"
    )
    for i in range(len(psi2) - 1):
        ax.plot(t2[i:i+2], psi2[i:i+2], color=colors2[i], lw=0.8)
    if trans_i:
        ax.axvline(t2[trans_i], color="purple", lw=1.2, ls=":",
                   label=f"Transition t≈{t2[trans_i]:.0f}")
    ax.axhline(THR_LR,  color="red",  lw=0.8, ls="--", alpha=0.6)
    ax.axhline(THR_SUP, color="blue", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("t (min)", fontsize=8)
    ax.set_ylabel("ψ (CHI)", fontsize=8)
    ax.set_title("Path 2: Suppress→Late-rise escape\n"
                 f"(A={A2}, D={D2})", fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1E88E5", label="Stage 1 (Suppress)"),
        Patch(color="#E53935", label="Stage 2 (Late-rise)"),
    ], fontsize=6, loc="upper right")
    if trans_i:
        ax.legend(fontsize=6)

    # ── Bottom-left: τ vs A ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    A_vals   = np.linspace(0.05, 0.50, 200)
    D_levels = [0.04, 0.06, 0.08, 0.12]
    colors_d = ["#1565C0", "#2E7D32", "#F57F17", "#B71C1C"]

    for d_val, col in zip(D_levels, colors_d):
        tau_arr = [kramers_escape_time(a, d_val) for a in A_vals]
        ax.semilogy(A_vals, tau_arr, color=col, lw=1.6, label=f"D={d_val}")

    # Overlay empirical data points (from Markov matrix inversion; qualitative)
    for state in ["Suppress", "Late-rise"]:
        A_state  = STATE_A[state]
        if A_state <= 0:
            continue
        p_emp    = MARKOV_SELF_TRANS[state]
        # Invert P_self = exp(-window/τ)  →  τ = -window / ln(P)
        tau_emp  = -WINDOW_SEC / np.log(p_emp + 1e-12)
        ax.plot(A_state, tau_emp, "k^", ms=8, zorder=5,
                label=f"Empirical τ ({state})" if state == "Suppress" else "")
        ax.annotate(state, (A_state, tau_emp), fontsize=6,
                    xytext=(5, 2), textcoords="offset points")

    ax.set_xlabel("A (control parameter)", fontsize=8)
    ax.set_ylabel("Kramers escape time τ (min)", fontsize=8)
    ax.set_title("τ vs A  (four noise levels)", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)
    ax.text(0.05, 0.05,
            "Large-noise caveat:\nΔV/D ≈ 0.17–0.33\n(Kramers: qualitative only)",
            transform=ax.transAxes, fontsize=6, va="bottom",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    # ── Bottom-centre: predicted vs observed P_self ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    states_plot = ["Suppress", "Late-rise", "Stable", "Marginal+"]
    A_states    = [STATE_A[s]      for s in states_plot]
    P_emp       = [MARKOV_SELF_TRANS[s] for s in states_plot]
    P_pred      = [kramers_self_transition_prob(a, D, WINDOW_SEC)
                   for a in A_states]

    x = np.arange(len(states_plot))
    w = 0.32
    bars_emp  = ax.bar(x - w/2, P_emp,  w, label="Empirical (Paper 1)",
                       color="#546E7A", alpha=0.85)
    bars_pred = ax.bar(x + w/2, P_pred, w, label="Kramers (additive)",
                       color="#EF5350", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(states_plot, fontsize=7)
    ax.set_ylabel("Self-transition probability", fontsize=8)
    ax.set_title("Predicted vs observed P_self", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6)

    # Annotate discrepancy for Suppress and Late-rise
    for i, state in enumerate(states_plot[:2]):
        disc = P_emp[i] - P_pred[i]
        ax.annotate(f"Δ={disc:+.2f}", xy=(x[i], max(P_emp[i], P_pred[i]) + 0.02),
                    ha="center", fontsize=6, color="black")

    ax.text(0.05, 0.95,
            "Directional ordering\nreproduced.\nQuantitative discrepancy\n"
            "→ multiplicative noise\n(Fig E)",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    # ── Bottom-right: parameter summary table ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    table_data = [
        ["Parameter", "Value", "Source"],
        ["Rc",        "0.991", "Cross-cohort regression"],
        ["a₀",        "20.5",  "Regression slope"],
        ["B",         "1.0",   "Normalised"],
        ["D",         "0.06±0.02", "FP KL-minimisation"],
        ["β (emp.)",  "0.88",  "PV prevalence scaling"],
        ["β (theo.)", "0.50",  "Pitchfork (theoretical)"],
        ["κ (mult.)", "1.5",   "Multiplicative noise"],
        ["ε₀ (mult.)","0.346", "D_eff = 0.06 target"],
        ["ΔV/D",      "0.17–0.33", "Large-noise regime"],
    ]
    col_widths = [0.30, 0.30, 0.40]
    tab = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1.0, 1.35)
    ax.set_title("Parameter estimation summary", fontsize=8, fontweight="bold",
                 pad=10)

    fig.suptitle(
        "Figure D  |  Kramers Escape-Time vs Empirical Self-Transitions\n"
        "[All estimates exploratory. Kramers predictions are qualitative ordering "
        "only (large-noise regime). SEED=42]",
        fontsize=9
    )

    out_path = os.path.join(OUTPUT_DIR, "fig_D_kramers.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
