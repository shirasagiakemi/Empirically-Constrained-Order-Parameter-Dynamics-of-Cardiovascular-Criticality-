"""
fig_F_psi_reconstruction.py
===========================
Reproduces Figure F of ECSoC Paper 2.

Order parameter ψ(t) reconstruction from individual RR interval time series.
Subject 0003 from the Spontaneous Ventricular Tachyarrhythmia Database (SVTDB),
PhysioNet (https://physionet.org/content/vt-db/1.0/).

Data required
-------------
Download the SVTDB zip from PhysioNet and place the extracted folder at:
  ./data/spontaneous-ventricular-tachyarrhythmia-database-1.0/

Records used:
  Pre-VT : 0003_mr4.qrs  +  0003_mr5.qrs  (concatenated; N=1,119 beats)
  VT     : 0003_vt3.qrs                    (N=861 beats)

Panel layout
------------
Top (full width) : ψ(t) = CHI(t) = 2(α₁ − α₂) time series with state
                   colour-coding and Pre-VT / VT boundary
Middle-left      : α₁(t) and α₂(t) time series
Middle-right     : R²(t) with Phase V threshold (R² < 0.93)
Bottom-left      : ψ distribution: Pre-VT vs VT
Bottom-right     : Summary statistics table

Methodology
-----------
ψ(t) is computed via sliding-window DFA:
  - Window  : 128 beats
  - Step    : 16 beats
  - α₁      : DFA exponent at short scales (4–16 beats)
  - α₂      : DFA exponent at long  scales (16–64 beats)
  - CHI(t)  : 2(α₁ − α₂)   [= ψ in the reduced order parameter equation]
  - R²(t)   : log-log scaling fit quality (16–64 beats)

ECSoC state thresholds (Paper 1 definitions)
  Suppress  : ψ < −0.05
  Marginal+ : −0.05 ≤ ψ < 0.148
  Late-rise : ψ ≥ 0.148
  Phase V   : R² < 0.93

Important caveats
-----------------
- N = 1 case study; no statistical inference is drawn.
- Pre-VT records (mr4, mr5) are non-contiguous clinical segments concatenated
  for visualization; discontinuity artefacts at the junction are possible.
- Window length (128 beats) limits α estimation at short records.
- SVTDB subjects are high-risk patients; Phase V prevalence >80% reflects
  severe baseline disease, not a universal pre-VT signature.
- All results are hypothesis-generating; prospective validation in
  time-resolved Holter cohorts is required.

Usage
-----
    python fig_F_psi_reconstruction.py [--data-dir PATH]

    Default data-dir: ./data/spontaneous-ventricular-tachyarrhythmia-database-1.0/data/

Output : ./output/fig_F_psi_reconstruction.pdf
         ./output/fig_F_psi_reconstruction.png

Reference
---------
Okabe H. Empirically Constrained Order Parameter Dynamics of Cardiovascular
Criticality: A Synergetics-Based Langevin Model of Arrhythmic Collapse.
Chaos (under review), 2026.

SVTDB: Goldberger AL et al. PhysioBank, PhysioToolkit, PhysioNet.
Circulation. 2000;101(23):e215-e220.
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# ECSoC thresholds (Paper 1 definitions)
# =============================================================================
THR_SUP   = -0.05   # ψ < THR_SUP  → Suppress
THR_LR    =  0.148  # ψ ≥ THR_LR   → Late-rise
PV_THRESH =  0.93   # R² < PV_THRESH → Phase V

# DFA sliding-window parameters
WIN      = 128   # beats
STEP     = 16    # beats
FS       = 1000  # SVTDB sampling rate (Hz)
RR_MIN   = 0.3   # physiological RR filter (s)
RR_MAX   = 2.0


# =============================================================================
# I/O helpers
# =============================================================================

def read_qrs(path: str, fs: float = FS) -> np.ndarray:
    """
    Read a SVTDB binary QRS position file.

    The file contains 16-bit unsigned integers representing R-peak sample
    positions at `fs` Hz. Consecutive differences give RR intervals (s).
    A physiological filter [RR_MIN, RR_MAX] removes ectopic/artefact beats.

    Parameters
    ----------
    path : str   Full path to the .qrs file.
    fs   : float Sampling frequency (default 1000 Hz for SVTDB).

    Returns
    -------
    rr : np.ndarray   RR intervals in seconds (filtered).
    """
    raw = np.fromfile(path, dtype=np.uint16)
    rr  = np.diff(raw.astype(np.float64)) / fs
    return rr[(rr > RR_MIN) & (rr < RR_MAX)]


# =============================================================================
# DFA
# =============================================================================

def dfa_window(rr: np.ndarray,
               scale_min: int = 4,
               scale_max: int = 64,
               n_scales: int  = 20) -> tuple:
    """
    Detrended Fluctuation Analysis on a single RR window.

    Returns (α₁, α₂, R²_short, R²_long).
    α₁ : DFA exponent at scales 4–16 beats  (short-range correlations)
    α₂ : DFA exponent at scales 16–64 beats (long-range correlations)
    R²  : log-log fit quality for the corresponding scale range

    All values are np.nan if the window is too short or fitting fails.

    Notes
    -----
    - Detrending uses degree-1 polynomial (linear) within each non-overlapping
      segment, following the original DFA formulation (Peng et al. 1995).
    - Scales are log-spaced; duplicates removed by np.unique.
    - Minimum 3 scale points required per fit; minimum N ≥ 2*scale_max beats.
    """
    N = len(rr)
    if N < scale_max * 2:
        return np.nan, np.nan, np.nan, np.nan

    # Integrated (profile) series
    y = np.cumsum(rr - rr.mean())

    # Scale grid
    scales = np.unique(np.round(
        np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
    ).astype(int))
    scales = scales[(scales >= scale_min) & (scales <= min(scale_max, N // 4))]

    # Fluctuation function F(s)
    F = []
    for s in scales:
        n_seg = N // s
        if n_seg < 2:
            F.append(np.nan)
            continue
        rms_list = []
        x_fit = np.arange(s, dtype=np.float64)
        for k in range(n_seg):
            seg   = y[k * s:(k + 1) * s]
            coef  = np.polyfit(x_fit, seg, 1)
            trend = np.polyval(coef, x_fit)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        F.append(float(np.mean(rms_list)))

    scales = np.array(scales, dtype=np.float64)
    F      = np.array(F,      dtype=np.float64)
    ok     = np.isfinite(F) & (F > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan

    log_s = np.log10(scales[ok])
    log_F = np.log10(F[ok])

    def _fit(mask: np.ndarray) -> tuple:
        """Linear fit on log-log; returns (slope, R²)."""
        if mask.sum() < 3:
            return np.nan, np.nan
        coef  = np.polyfit(log_s[mask], log_F[mask], 1)
        resid = log_F[mask] - np.polyval(coef, log_s[mask])
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((log_F[mask] - log_F[mask].mean()) ** 2)
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
        return float(coef[0]), float(r2)

    mask_short = (scales[ok] >= 4)  & (scales[ok] <= 16)
    mask_long  = (scales[ok] >= 16) & (scales[ok] <= 64)

    a1, r2_short = _fit(mask_short)
    a2, r2_long  = _fit(mask_long)
    return a1, a2, r2_short, r2_long


def sliding_dfa(rr_all: np.ndarray,
                win: int  = WIN,
                step: int = STEP) -> np.ndarray:
    """
    Apply DFA over a sliding window.

    Returns
    -------
    results : np.ndarray, shape (M, 5)
        Columns: [beat_centre, α₁, α₂, CHI, R²_long]
        Only rows with all-finite values are returned.
    """
    rows = []
    for start in range(0, len(rr_all) - win, step):
        seg = rr_all[start:start + win]
        a1, a2, _, r2_long = dfa_window(seg)
        chi = 2.0 * (a1 - a2) if (np.isfinite(a1) and np.isfinite(a2)) else np.nan
        rows.append((start + win // 2, a1, a2, chi, r2_long))

    arr   = np.array(rows)
    valid = np.all(np.isfinite(arr), axis=1)
    return arr[valid]


# =============================================================================
# State classification
# =============================================================================

def classify_chi(chi_val: float) -> tuple:
    """Return (state_name, hex_colour) for a single CHI value."""
    if chi_val >= THR_LR:
        return "Late-rise",  "#E53935"
    elif chi_val < THR_SUP:
        return "Suppress",   "#1E88E5"
    else:
        return "Marginal+",  "#8E24AA"


def majority_state(state_list: list) -> str:
    return Counter(state_list).most_common(1)[0][0]


# =============================================================================
# Plotting
# =============================================================================

def make_figure(beat_idx, alpha1, alpha2, chi, r2,
                boundary: int, subject: str = "0003") -> plt.Figure:

    pre_mask = beat_idx < boundary
    vt_mask  = ~pre_mask

    chi_pre = chi[pre_mask];  chi_vt = chi[vt_mask]
    r2_pre  = r2[pre_mask];   r2_vt  = r2[vt_mask]
    a1_pre  = alpha1[pre_mask]; a1_vt = alpha1[vt_mask]
    a2_pre  = alpha2[pre_mask]; a2_vt = alpha2[vt_mask]
    pv_pre  = np.mean(r2_pre < PV_THRESH)
    pv_vt   = np.mean(r2_vt  < PV_THRESH)

    states      = [classify_chi(c) for c in chi]
    state_names = [s[0] for s in states]
    state_cols  = [s[1] for s in states]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                            height_ratios=[2.2, 1.3, 1.3])

    # ── Panel 1: ψ(t) time series ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(len(chi) - 1):
        ax1.plot(beat_idx[i:i+2], chi[i:i+2],
                 color=state_cols[i], lw=1.5, alpha=0.85)

    ax1.axhline(THR_LR,  color="#E53935", lw=1.0, ls="--", alpha=0.7)
    ax1.axhline(THR_SUP, color="#1E88E5", lw=1.0, ls="--", alpha=0.7)
    ax1.axhline(0,        color="gray",   lw=0.5, ls=":")
    ax1.axvline(boundary, color="black",  lw=2.0, ls="-",  alpha=0.9)
    ax1.fill_betweenx([-2.8, 2.8], boundary, beat_idx.max(),
                      alpha=0.07, color="red")

    ax1.text(boundary + 12, 2.0, "VT\nperiod",
             fontsize=9, color="darkred", fontweight="bold")
    ax1.text(beat_idx.min() + 10, 2.0, "Pre-VT\n(sinus rhythm)",
             fontsize=9, color="#333333", fontweight="bold")

    ax1.set_xlabel("Beat index", fontsize=10)
    ax1.set_ylabel("ψ(t)  =  CHI(t)  =  2(α₁ − α₂)", fontsize=10)
    ax1.set_title(
        f"Figure F  |  Order Parameter ψ(t) Reconstruction — Individual Time Series\n"
        f"SVTDB Subject {subject}: Pre-VT → VT transition"
        f"  [Exploratory; N=1 case study]",
        fontsize=10, fontweight="bold"
    )
    ax1.set_xlim(beat_idx.min(), beat_idx.max())
    ax1.set_ylim(-2.8, 2.8)

    legend_patches = [
        mpatches.Patch(color="#E53935", label=f"Late-rise (ψ ≥ {THR_LR})"),
        mpatches.Patch(color="#8E24AA", label=f"Marginal+ ({THR_SUP} ≤ ψ < {THR_LR})"),
        mpatches.Patch(color="#1E88E5", label=f"Suppress (ψ < {THR_SUP})"),
        plt.Line2D([0],[0], color="black", lw=2,
                   label=f"VT onset (beat {boundary})"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left", ncol=2)

    # ── Panel 2: α₁ and α₂ ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(beat_idx, alpha1, color="#F57F17", lw=1.4,
             label="α₁ (4–16 beats)")
    ax2.plot(beat_idx, alpha2, color="#1565C0", lw=1.4,
             label="α₂ (16–64 beats)")
    ax2.axvline(boundary, color="black", lw=1.5, ls="-", alpha=0.7)
    ax2.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax2.set_xlabel("Beat index", fontsize=9)
    ax2.set_ylabel("DFA exponent", fontsize=9)
    ax2.set_title("α₁ and α₂ time series", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.set_xlim(beat_idx.min(), beat_idx.max())

    # ── Panel 3: R²(t) ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    r2_cols = np.where(r2 < PV_THRESH, "#E53935", "#43A047")
    for i in range(len(r2) - 1):
        ax3.plot(beat_idx[i:i+2], r2[i:i+2],
                 color=r2_cols[i], lw=1.4, alpha=0.85)
    ax3.axhline(PV_THRESH, color="red", lw=1.2, ls="--", alpha=0.8,
                label=f"Phase V threshold (R²={PV_THRESH})")
    ax3.axvline(boundary,  color="black", lw=1.5, ls="-", alpha=0.7)
    ax3.set_xlabel("Beat index", fontsize=9)
    ax3.set_ylabel("R² (log-log scaling fit)", fontsize=9)
    ax3.set_title("Scaling fit quality R²(t)", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.set_xlim(beat_idx.min(), beat_idx.max())
    ax3.set_ylim(0, 1.05)
    ax3.text(0.02, 0.05,
             f"Overall Phase V rate: {np.mean(r2 < PV_THRESH):.1%}",
             transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    # ── Panel 4: ψ distribution ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(-2.8, 2.8, 32)
    ax4.hist(chi[pre_mask], bins=bins, alpha=0.55, color="#1565C0",
             density=True, label=f"Pre-VT  (N={pre_mask.sum()} windows)")
    ax4.hist(chi[vt_mask],  bins=bins, alpha=0.55, color="#E53935",
             density=True, label=f"VT  (N={vt_mask.sum()} windows)")
    ax4.axvline(THR_LR,  color="#E53935", lw=1.0, ls="--", alpha=0.7)
    ax4.axvline(THR_SUP, color="#1E88E5", lw=1.0, ls="--", alpha=0.7)
    ax4.set_xlabel("ψ (CHI)", fontsize=9)
    ax4.set_ylabel("Density", fontsize=9)
    ax4.set_title("ψ distribution: Pre-VT vs VT", fontsize=9,
                  fontweight="bold")
    ax4.legend(fontsize=7)

    # ── Panel 5: summary table ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    state_pre_list = [state_names[i] for i in range(len(state_names))
                      if pre_mask[i]]
    state_vt_list  = [state_names[i] for i in range(len(state_names))
                      if vt_mask[i]]

    table_data = [
        ["Metric",         "Pre-VT",                              "VT (vt3)"],
        ["N (windows)",    str(pre_mask.sum()),                   str(vt_mask.sum())],
        ["CHI mean ± SD",
         f"{chi_pre.mean():.3f} ± {chi_pre.std():.3f}",
         f"{chi_vt.mean():.3f}  ± {chi_vt.std():.3f}"],
        ["α₁ mean",        f"{a1_pre.mean():.3f}",               f"{a1_vt.mean():.3f}"],
        ["α₂ mean",        f"{a2_pre.mean():.3f}",               f"{a2_vt.mean():.3f}"],
        ["R² mean",        f"{r2_pre.mean():.3f}",               f"{r2_vt.mean():.3f}"],
        ["Phase V rate",   f"{pv_pre:.1%}",                      f"{pv_vt:.1%}"],
        ["Dominant state", majority_state(state_pre_list),        majority_state(state_vt_list)],
    ]

    tab = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.5)
    ax5.set_title("Summary statistics", fontsize=9,
                  fontweight="bold", pad=10)

    # ── Footer caveat ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        f"[Exploratory — N=1 case study. Window={WIN} beats, step={STEP} beats. "
        f"DFA: α₁ (4–16 beats), α₂ (16–64 beats), R² (16–64 beats log-log fit). "
        f"ψ(t) ≡ CHI(t) = 2(α₁−α₂). "
        f"Pre-VT: mr4+mr5 concatenated. VT: vt3. SVTDB Subject {subject}.]",
        ha="center", fontsize=7, color="gray"
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure F: ψ(t) reconstruction from SVTDB Subject 0003."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/spontaneous-ventricular-tachyarrhythmia-database-1.0/data/",
        help="Path to SVTDB data directory containing .qrs files."
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    # ── Load data ─────────────────────────────────────────────────────────
    records_pre = ["0003_mr4", "0003_mr5"]
    record_vt   = "0003_vt3"

    print("Loading SVTDB Subject 0003...")
    for rec in records_pre + [record_vt]:
        path = os.path.join(data_dir, rec + ".qrs")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            print("Download SVTDB from https://physionet.org/content/vt-db/1.0/")
            sys.exit(1)

    rr_pre_parts = [read_qrs(os.path.join(data_dir, r + ".qrs"))
                    for r in records_pre]
    rr_pre = np.concatenate(rr_pre_parts)
    rr_vt  = read_qrs(os.path.join(data_dir, record_vt + ".qrs"))
    rr_all = np.concatenate([rr_pre, rr_vt])
    boundary = len(rr_pre)

    print(f"Pre-VT (mr4+mr5) : N={len(rr_pre)} beats, "
          f"mean RR={rr_pre.mean():.3f} s")
    print(f"VT (vt3)          : N={len(rr_vt)} beats, "
          f"mean RR={rr_vt.mean():.3f} s")
    print(f"Total             : N={len(rr_all)} beats")

    # ── Sliding-window DFA ────────────────────────────────────────────────
    print(f"\nRunning sliding-window DFA (win={WIN}, step={STEP})...")
    results  = sliding_dfa(rr_all, WIN, STEP)
    beat_idx = results[:, 0]
    alpha1   = results[:, 1]
    alpha2   = results[:, 2]
    chi      = results[:, 3]
    r2       = results[:, 4]
    print(f"Valid windows: {len(results)}")

    pre_mask = beat_idx < boundary
    vt_mask  = ~pre_mask
    print(f"\nPre-VT: CHI={chi[pre_mask].mean():.3f} ± {chi[pre_mask].std():.3f}, "
          f"R²={r2[pre_mask].mean():.3f}, "
          f"Phase V={np.mean(r2[pre_mask] < PV_THRESH):.1%}")
    print(f"VT:     CHI={chi[vt_mask].mean():.3f}  ± {chi[vt_mask].std():.3f}, "
          f"R²={r2[vt_mask].mean():.3f}, "
          f"Phase V={np.mean(r2[vt_mask] < PV_THRESH):.1%}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = make_figure(beat_idx, alpha1, alpha2, chi, r2, boundary)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig_F_psi_reconstruction.pdf")
    png_path = os.path.join(out_dir, "fig_F_psi_reconstruction.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
