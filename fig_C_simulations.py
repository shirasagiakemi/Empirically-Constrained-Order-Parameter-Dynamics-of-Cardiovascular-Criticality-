"""
fig_C_simulations.py
====================
Reproduces Figure C of ECSoC Paper 2.

Stochastic Langevin simulations — four canonical trajectory classes.

Column 1 : Path 2 / Supercritical  (A=+0.25, ε=0.35)  Suppress→Late-rise
Column 2 : Metastable Supercritical (A=+0.10, ε=0.20)  healthy norm
Column 3 : Path 1 / Suppress        (A=−0.20, ε=0.25)  subcritical fixation
Column 4 : Near-critical / Marginal+ (A≈0,    ε=0.30)  flat potential

Row 1 : ψ(t) time series with state colour-coding
Row 2 : Reconstructed potential V(ψ) + empirical histogram
Row 3 : Phase space (ψ × φ) with nullclines

Integrator : Euler-Maruyama (additive noise; exact in distribution)
Random seed: SEED = 42

[Simulation; parameters estimated from Paper 1; not a fit to individual
records. See manuscript Section 4.]

Usage
-----
    python fig_C_simulations.py
Output : ./output/fig_C_simulations.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

from langevin_core import (
    B, D, euler_maruyama, potential, fp_stationary,
    classify_state, state_fractions,
    THR_SUP, THR_LR
)

SEED = 42
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Four canonical parameter sets ────────────────────────────────────────────
CASES = [
    dict(label="Path 2 / Supercritical",
         A=+0.25, D_noise=0.35**2 / 2,   # ε=0.35 → D=ε²/2
         psi0=-0.30, n_steps=3000, dt=0.1,
         color="#1565C0"),
    dict(label="Metastable Supercritical",
         A=+0.10, D_noise=0.20**2 / 2,
         psi0=+0.20, n_steps=3000, dt=0.1,
         color="#2E7D32"),
    dict(label="Path 1 / Suppress",
         A=-0.20, D_noise=0.25**2 / 2,
         psi0=-0.10, n_steps=3000, dt=0.1,
         color="#E65100"),
    dict(label="Near-critical / Marginal+",
         A=0.001, D_noise=0.30**2 / 2,
         psi0=0.00, n_steps=3000, dt=0.1,
         color="#6A1B9A"),
]

# State colour map
STATE_COLORS = {
    "Late-rise":  "#E53935",
    "Suppress":   "#1E88E5",
    "Stable":     "#43A047",
    "Marginal+":  "#8E24AA",
}

# Slow variable coupling (for phase-space; qualitative only)
LAMBDA_PHI = 0.8   # φ relaxation rate
K_COUPLING = 0.2   # ψ → φ coupling


def simulate_phi(psi: np.ndarray, dt: float,
                 lam: float = LAMBDA_PHI, k: float = K_COUPLING,
                 rng: np.random.Generator = None) -> np.ndarray:
    """Integrate dφ/dt = -λφ + kψ  (Euler; φ is fast-slaved variable)."""
    if rng is None:
        rng = np.random.default_rng()
    phi    = np.empty_like(psi)
    phi[0] = k / lam * psi[0]  # quasi-static initialisation
    for i in range(len(psi) - 1):
        phi[i+1] = phi[i] + (-lam * phi[i] + k * psi[i]) * dt
    return phi


def main():
    rng = np.random.default_rng(SEED)
    psi_grid = np.linspace(-1.5, 1.5, 500)

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.32)

    for col, case in enumerate(CASES):
        A        = case["A"]
        D_noise  = case["D_noise"]
        n_steps  = case["n_steps"]
        dt       = case["dt"]
        col_main = case["color"]

        psi    = euler_maruyama(case["psi0"], A, D_noise, n_steps, dt,
                                rng=rng)
        phi    = simulate_phi(psi, dt, rng=rng)
        t      = np.arange(n_steps + 1) * dt
        states = classify_state(psi)
        fracs  = state_fractions(states)

        # ── Row 1: time series ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, col])
        for seg_start in range(len(psi) - 1):
            color = STATE_COLORS.get(states[seg_start], "gray")
            ax1.plot(t[seg_start:seg_start+2], psi[seg_start:seg_start+2],
                     color=color, lw=0.7)
        ax1.axhline(THR_LR,  color="red",  lw=0.8, ls="--", alpha=0.6)
        ax1.axhline(THR_SUP, color="blue", lw=0.8, ls="--", alpha=0.6)
        ax1.axhline(0,       color="gray", lw=0.5, ls=":")
        ax1.set_title(case["label"], fontsize=8, fontweight="bold")
        ax1.set_xlabel("t (min)", fontsize=7)
        if col == 0:
            ax1.set_ylabel("ψ (CHI)", fontsize=7)
        # State fraction text
        frac_text = "\n".join(
            f"{s}: {fracs[s]:.2f}" for s in
            ["Late-rise", "Suppress", "Stable", "Marginal+"]
        )
        ax1.text(0.98, 0.98, frac_text, transform=ax1.transAxes,
                 fontsize=5.5, va="top", ha="right",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.75))

        # ── Row 2: potential + histogram ──────────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        V   = potential(psi_grid, A)
        V  -= V.min()           # shift minimum to 0 for visual clarity
        ax2.plot(psi_grid, V, color=col_main, lw=2.0, label="V(ψ)")

        # Empirical histogram of simulated ψ (normalised to match V scale)
        counts, edges = np.histogram(psi, bins=60, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        scale   = V.max() / (counts.max() + 1e-12) * 0.6
        ax2.fill_between(centres, counts * scale, alpha=0.35,
                         color=col_main, step="mid")

        # Fixed points
        ax2.axvline(0, color="gray", lw=0.6, ls=":")
        if A > 0:
            psi_star = np.sqrt(A / B)
            for ps in [-psi_star, psi_star]:
                ax2.plot(ps, potential(np.array([ps]), A)[0] - V.min(),
                         "o", color=col_main, ms=6, zorder=5)
        ax2.plot(0, 0, "o", ms=6, zorder=5,
                 color="gray" if A > 0 else col_main,
                 mfc="white" if A > 0 else col_main)

        ax2.set_xlabel("ψ", fontsize=7)
        if col == 0:
            ax2.set_ylabel("V(ψ)  [+histogram]", fontsize=7)
        ax2.text(0.05, 0.95, f"A = {A:+.3f}",
                 transform=ax2.transAxes, fontsize=7, va="top",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # ── Row 3: phase space ψ × φ ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[2, col])
        n_pts  = len(psi)
        colors = cm.plasma(np.linspace(0.15, 0.95, n_pts))
        ax3.scatter(psi[::4], phi[::4], c=colors[::4], s=0.4, alpha=0.6,
                    rasterized=True)

        # Nullclines
        psi_nc = np.linspace(psi.min() - 0.1, psi.max() + 0.1, 200)
        # dψ/dt = 0: A*ψ - B*ψ³ = 0  →  φ = -(A*ψ - B*ψ³)/c  (c≈0; qualitative)
        phi_nc_psi = np.zeros_like(psi_nc)          # approximate nullcline
        phi_nc_phi = K_COUPLING / LAMBDA_PHI * psi_nc  # dφ/dt = 0: φ = k/λ * ψ
        ax3.plot(psi_nc, phi_nc_psi, "r-",  lw=1.0, alpha=0.7,
                 label="dψ/dt=0")
        ax3.plot(psi_nc, phi_nc_phi, "b--", lw=1.0, alpha=0.7,
                 label="dφ/dt=0")

        ax3.set_xlabel("ψ", fontsize=7)
        if col == 0:
            ax3.set_ylabel("φ (ΔP(III) analogue)", fontsize=7)
        ax3.legend(fontsize=5.5, loc="upper right")

    # ── Legend for state colours ──────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, lw=2, label=s)
        for s, c in STATE_COLORS.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Figure C  |  Stochastic Langevin Simulations — Four Trajectory Classes\n"
        "[Simulation; parameters estimated from Paper 1; not a fit to individual "
        "records. SEED=42]",
        fontsize=9
    )

    out_path = os.path.join(OUTPUT_DIR, "fig_C_simulations.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
