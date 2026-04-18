"""
fig_A_fokker_planck.py
======================
Reproduces Figure A of ECSoC Paper 2.

Fokker-Planck potential reconstruction from cross-cohort CHI distributions.
Four representative cohorts (NSR Healthy, SVTDB Pre, SVTDB VT, MVEDB) spanning
the full disease spectrum.

Upper row : reconstructed potential V(ψ) = -(A/2)ψ² + (B/4)ψ⁴
Lower row : Fokker-Planck p_st (black line) overlaid on empirical CHI
            distribution (Gaussian approximation; coloured fill)

[Exploratory; parameter values are initial estimates requiring time-series
validation. See manuscript Section 6.3 for limitations of Gaussian
approximation.]

Usage
-----
    python fig_A_fokker_planck.py
Output : ./output/fig_A_fokker_planck.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

from langevin_core import (
    COHORT_DATA, control_A, potential, fp_stationary, fit_D_kl
)

# Reproducibility (parameter estimation uses deterministic optimisation;
# no stochastic elements in this figure)
RNG_SEED = 42

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cohorts to plot ───────────────────────────────────────────────────────────
PLOT_COHORTS = ["NSR Healthy", "SVTDB Pre", "SVTDB VT", "MVEDB"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
PSI_GRID = np.linspace(-1.8, 1.8, 600)


def main():
    fig = plt.figure(figsize=(13, 7))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for col, (cohort, color) in enumerate(zip(PLOT_COHORTS, COLORS)):
        chi_mean, chi_sd, R2_mean, phaseV, N = COHORT_DATA[cohort]
        A = control_A(R2_mean)
        D_fit = fit_D_kl(chi_mean, chi_sd, A)

        V     = potential(PSI_GRID, A)
        p_fp  = fp_stationary(PSI_GRID, A, D_fit)
        p_emp = norm.pdf(PSI_GRID, loc=chi_mean, scale=chi_sd)
        p_emp /= np.trapezoid(p_emp, PSI_GRID)

        # ── Top row: potential ────────────────────────────────────────────
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.plot(PSI_GRID, V, color=color, lw=2)
        ax_top.axhline(0, color="gray", lw=0.6, ls="--")
        ax_top.axvline(0, color="gray", lw=0.6, ls=":")
        ax_top.set_xlim(-1.6, 1.6)

        # Mark fixed points
        if A > 0:
            psi_star = np.sqrt(A)
            for ps in [-psi_star, psi_star]:
                ax_top.plot(ps, potential(np.array([ps]), A)[0],
                            "o", color=color, ms=6, zorder=5)
        ax_top.plot(0, 0, "o", color="gray" if A > 0 else color, ms=6,
                    mfc="white" if A > 0 else color, zorder=5)

        ax_top.set_title(cohort, fontsize=9, fontweight="bold")
        ax_top.set_xlabel("ψ (CHI)", fontsize=8)
        if col == 0:
            ax_top.set_ylabel("V(ψ)", fontsize=8)
        ax_top.text(0.05, 0.95, f"A = {A:+.3f}\nR² = {R2_mean:.3f}",
                    transform=ax_top.transAxes, va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        # ── Bottom row: Fokker-Planck fit ─────────────────────────────────
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.fill_between(PSI_GRID, p_emp, alpha=0.35, color=color,
                            label="Empirical (Gaussian approx.)")
        ax_bot.plot(PSI_GRID, p_fp, "k-", lw=1.8,
                    label=f"FP fit  D={D_fit:.3f}")
        ax_bot.set_xlim(-1.6, 1.6)
        ax_bot.set_xlabel("ψ (CHI)", fontsize=8)
        if col == 0:
            ax_bot.set_ylabel("Probability density", fontsize=8)
        ax_bot.legend(fontsize=6, loc="upper right")
        ax_bot.text(0.05, 0.95, f"D_fit = {D_fit:.3f}",
                    transform=ax_bot.transAxes, va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Figure-level annotation ───────────────────────────────────────────────
    fig.suptitle(
        "Figure A  |  Fokker-Planck Potential Reconstruction\n"
        "[Exploratory — parameter values are initial estimates requiring "
        "time-series validation]",
        fontsize=9, y=1.01
    )

    out_path = os.path.join(OUTPUT_DIR, "fig_A_fokker_planck.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
