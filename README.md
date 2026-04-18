# ECSoC Paper 2 — Simulation Code

**Empirically Constrained Order Parameter Dynamics of Cardiovascular Criticality:  
A Synergetics-Based Langevin Model of Arrhythmic Collapse**

Hiroyuki Okabe, Independent Researcher, Tokyo, Japan

---

## Overview

This repository contains all simulation and analysis code for Paper 2 of the ECSoC series.
The code reproduces Figures A–E of the manuscript:

| Script | Figure | Content |
|--------|--------|---------|
| `fig_A_fokker_planck.py` | Figure A | Fokker-Planck potential reconstruction from CHI distributions |
| `fig_B_crosscohort.py` | Figure B | Cross-cohort A(R²) derivation and bifurcation diagram |
| `fig_C_simulations.py` | Figure C | Stochastic Langevin simulations — four trajectory classes |
| `fig_D_kramers.py` | Figure D | Kramers escape-time vs empirical self-transitions |
| `fig_E_multiplicative.py` | Figure E | Multiplicative noise analytical predictions |
| `langevin_core.py` | — | Shared simulation engine (imported by all figure scripts) |

All figures in the manuscript were generated from these scripts with the random seed
`SEED = 42` (set at the top of each simulation script).

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

No proprietary data are required. All parameters are estimated from the cross-cohort
summary statistics reported in Paper 1 (Okabe 2026) and are hard-coded in
`langevin_core.py` with full provenance comments.

---

## Quick start

```bash
git clone https://github.com/shirasagiakemi/Empirically-Constrained-Order-Parameter-Dynamics-of-Cardiovascular-Criticality-.git
cd Empirically-Constrained-Order-Parameter-Dynamics-of-Cardiovascular-Criticality-
python fig_C_simulations.py        # Reproduces Figure C
python fig_D_kramers.py            # Reproduces Figure D
python fig_A_fokker_planck.py      # Reproduces Figure A
python fig_B_crosscohort.py        # Reproduces Figure B
python fig_E_multiplicative.py     # Reproduces Figure E
```

Each script saves a high-resolution PDF to `./output/`.

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
- All simulation results are **hypothesis-generating**; prospective validation
  in time-resolved Holter cohorts is required.

---

## Citation

If you use this code, please cite:

> Okabe H. Empirically Constrained Order Parameter Dynamics of Cardiovascular
> Criticality: A Synergetics-Based Langevin Model of Arrhythmic Collapse.
> *Chaos* (under review), 2026.

---

## License

MIT License. See `LICENSE` for details.
