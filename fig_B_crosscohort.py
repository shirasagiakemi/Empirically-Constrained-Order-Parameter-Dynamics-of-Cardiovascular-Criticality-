"""
fig_B_crosscohort.py
====================
Reproduces Figure B of ECSoC Paper 2.

Cross-cohort A(R²) derivation and bifurcation diagram.

Left   : A(R²) estimation across 10 cohorts; colour = Phase V rate;
          size ∝ √N; linear fit A = a0*(R² - Rc)
Centre : Phase V prevalence vs (Rc - R²) on log-log axes; empirical β=0.88
          vs theoretical β=0.5
Right  : Bifurcation diagram ψ* vs A with empirical cohort overlay

[All estimates exploratory; cross-sectional cohort data only.
See manuscript Section 3.2 and 6.3.]

Usage
-----
    python fig_B_crosscohort.py
Output : ./output/fig_B_crosscohort.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import linregress

from langevin_core import (
    COHORT_DATA, Rc, a0, B,
    beta_empirical, beta_theoretical, control_A
)

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ── Unpack cohort data ────────────────────────────────────────────────────
    cohort_names = list(COHORT_DATA.keys())
    chi_means  = np.array([COHORT_DATA[c][0] for c in cohort_names])
    R2_means   = np.array([COHORT_DATA[c][2] for c in cohort_names])
    phaseV     = np.array([COHORT_DATA[c][3] for c in cohort_names])
    Ns         = np.array([COHORT_DATA[c][4] for c in cohort_names])

    # A estimated from fixed-point: ψ* ≈ CHI_mean, A ≈ sign(CHI)*CHI²
    A_est = np.sign(chi_means) * chi_means**2

    # Restrict to Phase V < 10% for regression (per manuscript)
    mask  = phaseV < 0.10
    slope_emp = a0   # from manuscript cross-cohort regression
    Rc_emp    = Rc

    R2_fit   = np.linspace(0.91, 1.00, 200)
    A_fit    = slope_emp * (R2_fit - Rc_emp)

    # ── Colour map: Phase V rate ──────────────────────────────────────────────
    norm_pv  = mcolors.Normalize(vmin=0, vmax=phaseV.max() + 0.05)
    cmap     = cm.RdYlGn_r

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Left: A vs R² ─────────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(R2_means, A_est,
                    c=phaseV, cmap=cmap, norm=norm_pv,
                    s=60 * np.sqrt(Ns / Ns.min()),
                    edgecolors="k", linewidths=0.5, zorder=3)
    ax.plot(R2_fit, A_fit, "k-", lw=1.6, label=f"A = {slope_emp}(R² − {Rc_emp})")
    ax.axhline(0, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.axvline(Rc_emp, color="red", ls=":", lw=1.0, alpha=0.7)

    for i, name in enumerate(cohort_names):
        ax.annotate(name, (R2_means[i], A_est[i]),
                    fontsize=5.5, ha="left", va="bottom",
                    xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel("R² (scaling fit quality)", fontsize=9)
    ax.set_ylabel("A (control parameter estimate)", fontsize=9)
    ax.set_title("Cross-cohort A(R²) estimation", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Phase V rate", fontsize=7)
    ax.text(0.05, 0.05,
            f"Rc = {Rc_emp}\na₀ = {slope_emp}\n[restricted to PV<10%]",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ── Centre: Phase V scaling (log-log) ────────────────────────────────────
    ax = axes[1]
    # Only cohorts with Phase V > 0 and R² < Rc
    mask_pv = (phaseV > 0) & (R2_means < Rc_emp)
    delta_R2 = Rc_emp - R2_means[mask_pv]
    pv_vals  = phaseV[mask_pv]

    ax.scatter(delta_R2, pv_vals, c=phaseV[mask_pv], cmap=cmap, norm=norm_pv,
               s=80, edgecolors="k", linewidths=0.5, zorder=3)

    # Fit lines
    x_line = np.linspace(delta_R2.min() * 0.8, delta_R2.max() * 1.2, 100)
    # Empirical β = 0.88; scale factor from first data point
    scale_emp  = pv_vals[0] / delta_R2[0]**beta_empirical
    scale_theo = pv_vals[0] / delta_R2[0]**beta_theoretical

    ax.plot(x_line, scale_emp  * x_line**beta_empirical,
            "r-",  lw=1.6, label=f"Empirical β = {beta_empirical}")
    ax.plot(x_line, scale_theo * x_line**beta_theoretical,
            "b--", lw=1.4, label=f"Theoretical β = {beta_theoretical}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rc − R²", fontsize=9)
    ax.set_ylabel("Phase V prevalence", fontsize=9)
    ax.set_title("Phase V scaling (log-log)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.text(0.05, 0.95,
            "[Exploratory;\ncross-sectional data]",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # ── Right: bifurcation diagram ψ* vs A ───────────────────────────────────
    ax = axes[2]
    A_range = np.linspace(-0.5, 0.5, 500)

    A_pos        = np.where(A_range > 0, A_range, 0.0)
    psi_stable   = np.where(A_range > 0,  np.sqrt(A_pos / B), np.nan)
    psi_stable_n = np.where(A_range > 0, -np.sqrt(A_pos / B), np.nan)
    psi_unstable = np.where(A_range > 0,  np.zeros_like(A_range), np.nan)
    psi_zero     = np.where(A_range <= 0, np.zeros_like(A_range), np.nan)

    ax.plot(A_range, psi_stable,    "b-",  lw=2,   label="Stable (ψ*>0)")
    ax.plot(A_range, psi_stable_n,  "b-",  lw=2)
    ax.plot(A_range, psi_unstable,  "b--", lw=1.2, label="Unstable (ψ*=0, A>0)")
    ax.plot(A_range, psi_zero,      "k-",  lw=2,   label="Stable (ψ*=0, A<0)")

    # Overlay empirical cohorts
    sc2 = ax.scatter(A_est, chi_means,
                     c=phaseV, cmap=cmap, norm=norm_pv,
                     s=60 * np.sqrt(Ns / Ns.min()),
                     edgecolors="k", linewidths=0.6, zorder=4,
                     label="Cohorts")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="red",  lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("A (control parameter)", fontsize=9)
    ax.set_ylabel("ψ* (fixed point / CHI_mean)", fontsize=9)
    ax.set_title("Bifurcation diagram", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Figure B  |  Cross-Cohort Parameter Estimation and Bifurcation Diagram\n"
        "[Exploratory — cross-sectional cohort data only]",
        fontsize=9
    )

    out_path = os.path.join(OUTPUT_DIR, "fig_B_crosscohort.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
