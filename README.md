# ECSoC Paper 2 — Simulation Code

**Empirically Constrained Order Parameter Dynamics of Cardiovascular Criticality:  
A Synergetics-Based Langevin Model of Arrhythmic Collapse**

Hiroyuki Okabe, Independent Researcher, Tokyo, Japan

---

## Overview

This repository contains all simulation and analysis code for Paper 2 of the ECSoC series.
The code reproduces Figures A–F of the manuscript:

| Script | Figure | Content |
|--------|--------|---------|
| `fig_A_fokker_planck.py` | Figure A | Fokker-Planck potential reconstruction from CHI distributions |
| `fig_B_crosscohort.py` | Figure B | Cross-cohort A(R²) derivation and bifurcation diagram |
| `fig_C_simulations.py` | Figure C | Stochastic Langevin simulations — four trajectory classes |
| `fig_D_kramers.py` | Figure D | Kramers escape-time vs empirical self-transitions |
| `fig_E_multiplicative.py` | Figure E | Multiplicative noise analytical predictions |
| `fig_F_psi_reconstruction.py` | Figure F | ψ(t) order parameter reconstruction from individual RR time series (SVTDB Subject 0003) |
| `langevin_core.py` | — | Shared simulation engine (imported by all figure scripts) |

Figures A–E are fully reproducible with random seed `SEED = 42` (no external data required).  
Figure F requires the SVTDB dataset (see **Data** section below).

---

## Requirements

```
python >= 3.9
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

---

## Data (Figure F only)

Figure F uses **SVTDB Subject 0003** from the Spontaneous Ventricular
Tachyarrhythmia Database (PhysioNet):

> Goldberger AL et al. PhysioBank, PhysioToolkit, PhysioNet.  
> *Circulation*. 2000;101(23):e215–e220.  
> https://physionet.org/content/vt-db/1.0/

Download and extract the zip, then place the `data/` folder at:

```
./data/spontaneous-ventricular-tachyarrhythmia-database-1.0/data/
```

Records used: `0003_mr4.qrs`, `0003_mr5.qrs` (Pre-VT, concatenated),
`0003_vt3.qrs` (VT period).

---

## Quick start

```bash
git clone https://github.com/shirasagiakemi/Empirically-Constrained-Order-Parameter-Dynamics-of-Cardiovascular-Criticality-.git
cd Empirically-Constrained-Order-Parameter-Dynamics-of-Cardiovascular-Criticality-

# Figures A–E (no data required)
python fig_C_simulations.py
python fig_D_kramers.py
python fig_A_fokker_planck.py
python fig_B_crosscohort.py
python fig_E_multiplicative.py

# Figure F (requires SVTDB; see Data section above)
python fig_F_psi_reconstruction.py
# or with custom path:
python fig_F_psi_reconstruction.py --data-dir /path/to/svtdb/data/
```

Each script saves a high-resolution PDF (and PNG for Figure F) to `./output/`.

---

## Parameter provenance

All model parameters are derived from Paper 1 cross-cohort data (no free fitting):

| Parameter | Value | Source |
|-----------|-------|--------|
| Rc | 0.991 | Cross-cohort A(R²) linear regression |
| a₀ | 20.5 | Slope of A vs R² across 10 cohorts |
| B | 1.0 | Normalised (sets CHI scale) |
| D | 0.06 ± 0.02 | Fokker-Planck KL-divergence minimisation |
| β | 0.88 | Empirical Phase V prevalence scaling exponent |
| κ (mult. noise) | 1.5 | Multiplicative noise shape parameter |
| ε₀ (mult. noise) | 0.346 | Multiplicative noise amplitude (D=0.06 target) |

---

## Important caveats (from manuscript Section 6.3)

- All parameters are estimated from **cross-sectional cohort summary statistics**,
  not individual patient time series.
- The Fokker-Planck fitting approximates empirical CHI distributions as Gaussian
  (mean and SD from Paper 1); tail behaviour is not directly constrained.
- Kramers escape-time predictions are applied in the **large-noise regime**
  (ΔV/D ≈ 0.17–0.33), where Kramers theory is an approximation.
  Predicted self-transition probabilities should be interpreted as qualitative
  ordering, not quantitative precision.
- Figure F is an **N=1 case study**; no statistical inference is drawn.
- All simulation and reconstruction results are **hypothesis-generating**;
  prospective validation in time-resolved Holter cohorts is required.

---

## Citation

If you use this code, please cite:

> Okabe H. Empirically Constrained Order Parameter Dynamics of Cardiovascular
> Criticality: A Synergetics-Based Langevin Model of Arrhythmic Collapse.
> *Frontiers* (under review), 2026.

For Figure F, additionally cite the SVTDB:

> Goldberger AL et al. PhysioBank, PhysioToolkit, PhysioNet.
> *Circulation*. 2000;101(23):e215–e220.

---

## License

MIT License. See `LICENSE` for details.
