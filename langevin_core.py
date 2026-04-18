"""
langevin_core.py
================
Shared simulation engine for ECSoC Paper 2.

Provides:
  - All empirically estimated model parameters (with provenance)
  - Cross-cohort summary statistics from Paper 1
  - Euler-Maruyama integrator for the 1-D reduced order parameter equation
  - Heun (Stratonovich) integrator for multiplicative noise
  - Kramers escape-time formula (additive and multiplicative noise)
  - Fokker-Planck stationary distribution and KL divergence

Reference
---------
Okabe H. Empirically Constrained Order Parameter Dynamics of Cardiovascular
Criticality: A Synergetics-Based Langevin Model of Arrhythmic Collapse.
Chaos (under review), 2026.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# =============================================================================
# 1.  MODEL PARAMETERS  (empirically estimated; see manuscript Section 3)
# =============================================================================

# -- Bifurcation control --
Rc   = 0.991   # Critical R² (bifurcation threshold); cross-cohort regression intercept
a0   = 20.5    # A(R²) coupling: A = a0*(R² - Rc); cross-cohort regression slope
B    = 1.0     # ψ⁴ coefficient; normalised to set CHI scale

# -- Noise --
D    = 0.06    # Additive noise intensity; Fokker-Planck KL-minimisation (population mean)
D_SD = 0.02    # Between-cohort SD of D

# -- Multiplicative noise extension (Section 6.2) --
kappa   = 1.5    # Shape parameter: σ(ψ) = ε0 * exp(-κψ²)
epsilon0 = 0.346  # Amplitude: chosen so that D_eff = ε0²/(2*mean) ≈ D = 0.06

# -- Slow R² dynamics (qualitative only) --
gamma   = 0.01   # R² relaxation rate  [arbitrary time units]
sigma_R = 0.002  # Noise on R² trajectory

# -- Phase V empirical scaling --
beta_empirical  = 0.88   # Empirical Phase V prevalence ∝ (Rc - R²)^β
beta_theoretical = 0.50  # Ideal pitchfork prediction

# =============================================================================
# 2.  CROSS-COHORT SUMMARY STATISTICS  (Paper 1, Table 2 / Figure 3)
#     Format: cohort_name -> (CHI_mean, CHI_SD, R2_mean, PhaseV_rate, N)
# =============================================================================

COHORT_DATA = {
    #                         CHI_mean   CHI_SD   R2_mean   PhaseV%   N
    "NSR Healthy":           ( 0.334,    0.18,    0.997,    0.000,   54),
    "CHF2 NYHA1-3":          (-0.030,    0.21,    0.985,    0.000,   27),
    "SVTDB Pre":             (-0.051,    0.22,    0.983,    0.000,  160),
    "SVTDB VT":              (-0.146,    0.24,    0.971,    0.038,  160),
    "CAST Post-MI":          (-0.180,    0.25,    0.965,    0.120,   75),
    "MUSIC":                 (-0.220,    0.26,    0.958,    0.210,  100),
    "SDDB":                  (-0.250,    0.27,    0.951,    0.280,   75),
    "CUDB Acute VF":         (-0.295,    0.28,    0.942,    0.000,   35),
    "BIDMC CHF":             (-0.310,    0.29,    0.938,    0.533,   15),
    "MVEDB":                 (-0.370,    0.31,    0.921,    0.909,   22),
}

# Empirical Markov self-transition probabilities (Paper 1 Table 4)
MARKOV_SELF_TRANS = {
    "Suppress":   0.70,
    "Late-rise":  0.58,
    "Stable":     0.06,
    "Marginal+":  0.07,
}

# State A-values used in simulations (estimated from CHI_mean fixed-point)
STATE_A = {
    "Suppress":   -0.20,   # monostable; ψ* = 0
    "Late-rise":  +0.15,   # bistable; dominant ψ* = +√(A/B)
    "Stable":      0.00,   # marginal
    "Marginal+":  +0.05,   # near-critical
}

# State CHI thresholds (Paper 1 definitions)
THR_SUP = -0.05    # ψ < THR_SUP  => Suppress
THR_LR  =  0.148   # ψ > THR_LR   => Late-rise
# |ψ| < THR_LR and ψ >= THR_SUP  => Stable / Marginal+ (further sub-divided)

# =============================================================================
# 3.  HELPER FUNCTIONS
# =============================================================================

def control_A(R2: float) -> float:
    """Effective control parameter A(R²) = a0*(R² - Rc)."""
    return a0 * (R2 - Rc)


def potential(psi: np.ndarray, A: float, B: float = B) -> np.ndarray:
    """Deterministic double-well potential V(ψ) = -(A/2)ψ² + (B/4)ψ⁴."""
    return -(A / 2.0) * psi**2 + (B / 4.0) * psi**4


def fp_stationary(psi: np.ndarray, A: float, D_noise: float,
                  B: float = B) -> np.ndarray:
    """
    Fokker-Planck stationary distribution (unnormalised):
      p_st(ψ) ∝ exp(-V(ψ) / D)
    Normalised over the provided psi grid.
    """
    V = potential(psi, A, B)
    p = np.exp(-V / D_noise)
    dp = np.trapezoid(p, psi)
    return p / dp


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(p || q); adds small epsilon for numerical stability."""
    eps = 1e-12
    p = np.clip(p, eps, None);  p /= p.sum()
    q = np.clip(q, eps, None);  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def fit_D_kl(chi_mean: float, chi_sd: float, A: float,
             psi_range: tuple = (-2.0, 2.0), n_grid: int = 500) -> float:
    """
    Estimate noise intensity D by minimising KL divergence between the
    empirical CHI distribution (Gaussian approx.) and the Fokker-Planck
    stationary distribution.

    Parameters
    ----------
    chi_mean, chi_sd : float
        Cohort CHI mean and SD (Paper 1 summary statistics).
    A : float
        Control parameter for this cohort.
    psi_range : tuple
        Grid range for ψ.
    n_grid : int
        Number of grid points.

    Returns
    -------
    D_fit : float
        Estimated noise intensity.

    Notes
    -----
    The Gaussian approximation of p_empirical misrepresents tails (see
    manuscript Section 6.3). D estimates should be interpreted at the
    population level, not per-patient.
    """
    psi = np.linspace(psi_range[0], psi_range[1], n_grid)
    p_emp = norm.pdf(psi, loc=chi_mean, scale=chi_sd)
    p_emp /= np.trapezoid(p_emp, psi)

    def objective(log_D):
        D_try = np.exp(log_D)
        p_fp  = fp_stationary(psi, A, D_try)
        return kl_divergence(p_emp, p_fp)

    result = minimize_scalar(objective, bounds=(-5.0, 1.0), method="bounded")
    return float(np.exp(result.x))


# =============================================================================
# 4.  KRAMERS ESCAPE-TIME  (Section 3.3 / 6.2)
# =============================================================================

def barrier_height(A: float, B: float = B) -> float:
    """
    ΔV = V(0) - V(ψ*) for A > 0  (double-well barrier height).
    ΔV = A² / (4B).

    NOTE: In the present parameterisation ΔV ≈ 0.01–0.02 and D = 0.06,
    giving ΔV/D ≈ 0.17–0.33. Kramers theory assumes ΔV/D >> 1 (small-noise
    regime); here it provides qualitative ordering only.
    """
    if A <= 0:
        return 0.0
    return A**2 / (4.0 * B)


def kramers_escape_time(A: float, D_noise: float, B: float = B) -> float:
    """
    Mean first-passage time between the two wells (Kramers formula):
      τ = (π / |A|) * exp(ΔV / D)

    Valid strictly for ΔV/D >> 1. For the present large-noise regime
    (ΔV/D ≈ 0.2) this gives qualitative ordering only.

    Returns np.inf if A <= 0 (no barrier).
    """
    if A <= 0:
        return np.inf
    dV = barrier_height(A, B)
    return (np.pi / abs(A)) * np.exp(dV / D_noise)


def kramers_self_transition_prob(A: float, D_noise: float,
                                 window_sec: float = 16.0,
                                 B: float = B) -> float:
    """
    Convert Kramers escape time τ to self-transition probability over a
    sliding window of `window_sec` seconds:
      P_self = exp(-window_sec / τ)

    This uses an exponential waiting-time model and inherits the large-noise
    caveat of kramers_escape_time().
    """
    tau = kramers_escape_time(A, D_noise, B)
    if np.isinf(tau):
        return 0.0
    return float(np.exp(-window_sec / tau))


# --- Multiplicative noise Kramers extension ---

def sigma_multiplicative(psi: float, eps0: float = epsilon0,
                          kap: float = kappa) -> float:
    """State-dependent noise amplitude σ(ψ) = ε₀ * exp(-κψ²)."""
    return eps0 * np.exp(-kap * psi**2)


def effective_potential_stratonovich(psi: np.ndarray, A: float,
                                     D_noise: float, B: float = B,
                                     eps0: float = epsilon0,
                                     kap: float = kappa) -> np.ndarray:
    """
    Stratonovich effective potential (Ito → Stratonovich correction):
      V_eff(ψ) = V(ψ) + D * κ * ψ²
    Multiplicative noise deepens the wells near the fixed points,
    prolonging Kramers escape time.
    """
    V = potential(psi, A, B)
    return V + D_noise * kap * psi**2


def kramers_self_transition_multiplicative(A: float, D_noise: float,
                                           window_sec: float = 16.0,
                                           B: float = B,
                                           eps0: float = epsilon0,
                                           kap: float = kappa) -> float:
    """
    Analytical Kramers P_self under multiplicative noise.
    Uses the effective barrier height from V_eff.

    Returns
    -------
    P_self : float
        Predicted self-transition probability (qualitative; large-noise caveat
        applies; see manuscript Section 6.3).
    """
    if A <= 0:
        return 0.0
    # Effective barrier height from V_eff
    psi_star = np.sqrt(A / B)
    dV_eff   = (D_noise * kap * psi_star**2) + barrier_height(A, B)
    tau_eff  = (np.pi / abs(A)) * np.exp(dV_eff / D_noise)
    return float(np.exp(-window_sec / tau_eff))


# =============================================================================
# 5.  LANGEVIN INTEGRATORS
# =============================================================================

def euler_maruyama(psi0: float, A: float, D_noise: float,
                   n_steps: int, dt: float = 0.1,
                   B: float = B,
                   rng: np.random.Generator = None) -> np.ndarray:
    """
    Euler-Maruyama integration of the reduced order parameter equation:
      dψ = (A*ψ - B*ψ³) dt + sqrt(2D) dW

    Parameters
    ----------
    psi0 : float   Initial condition.
    A    : float   Control parameter.
    D_noise : float   Noise intensity.
    n_steps : int  Number of time steps.
    dt   : float   Time step (arbitrary units; 1 unit ≈ 1 min in physiology).
    B    : float   ψ⁴ coefficient.
    rng  : numpy Generator   Random number generator (for reproducibility).

    Returns
    -------
    psi : np.ndarray  shape (n_steps+1,)

    Notes
    -----
    Euler-Maruyama is first-order accurate in the strong sense. For additive
    noise (constant diffusion coefficient) it is exact in distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    psi      = np.empty(n_steps + 1)
    psi[0]   = psi0
    sqrt2D   = np.sqrt(2.0 * D_noise * dt)
    noise    = rng.standard_normal(n_steps)
    for i in range(n_steps):
        drift   = (A * psi[i] - B * psi[i]**3) * dt
        psi[i+1] = psi[i] + drift + sqrt2D * noise[i]
    return psi


def heun_stratonovich(psi0: float, A: float, D_noise: float,
                      n_steps: int, dt: float = 0.1,
                      B: float = B,
                      eps0: float = epsilon0,
                      kap: float = kappa,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Heun method (Stratonovich interpretation) for multiplicative noise:
      dψ = (A*ψ - B*ψ³) dt + σ(ψ) ∘ dW
      σ(ψ) = ε₀ exp(-κψ²)

    Heun is the standard 2nd-order predictor-corrector for Stratonovich SDEs.

    Parameters
    ----------
    (same as euler_maruyama, plus eps0, kap for multiplicative noise)

    Returns
    -------
    psi : np.ndarray  shape (n_steps+1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    psi    = np.empty(n_steps + 1)
    psi[0] = psi0
    sqrtdt = np.sqrt(dt)
    noise  = rng.standard_normal(n_steps)

    def drift(x):
        return A * x - B * x**3

    def sigma(x):
        return eps0 * np.exp(-kap * x**2)

    for i in range(n_steps):
        dW  = sqrtdt * noise[i]
        # Predictor
        x_p = psi[i] + drift(psi[i]) * dt + sigma(psi[i]) * dW
        # Corrector (Stratonovich)
        psi[i+1] = (psi[i]
                    + 0.5 * (drift(psi[i]) + drift(x_p)) * dt
                    + 0.5 * (sigma(psi[i]) + sigma(x_p)) * dW)
    return psi


def simulate_path1_chronic(R2_start: float = 0.995,
                            R2_end: float   = 0.920,
                            n_steps: int    = 3000,
                            dt: float       = 0.1,
                            psi0: float     = 0.30,
                            rng: np.random.Generator = None) -> dict:
    """
    Path 1 simulation: chronic R² deterioration driving bifurcation.

    R²(t) decreases linearly from R2_start to R2_end over n_steps.
    A(t) = a0*(R²(t) - Rc) crosses zero when R²(t) = Rc = 0.991.

    Returns
    -------
    dict with keys: 'psi', 'R2', 'A', 't'
    """
    if rng is None:
        rng = np.random.default_rng()
    R2_arr = np.linspace(R2_start, R2_end, n_steps + 1)
    A_arr  = a0 * (R2_arr - Rc)
    psi    = np.empty(n_steps + 1)
    psi[0] = psi0
    sqrt2D = np.sqrt(2.0 * D * dt)
    noise  = rng.standard_normal(n_steps)
    for i in range(n_steps):
        drift    = (A_arr[i] * psi[i] - B * psi[i]**3) * dt
        psi[i+1] = psi[i] + drift + sqrt2D * noise[i]
    t = np.arange(n_steps + 1) * dt
    return {"psi": psi, "R2": R2_arr, "A": A_arr, "t": t}


def classify_state(psi: np.ndarray) -> np.ndarray:
    """
    Classify each ψ value into ECSoC state label.

    Returns
    -------
    states : np.ndarray of str, same shape as psi
      'Suppress'  : ψ < THR_SUP
      'Stable'    : THR_SUP <= ψ < 0
      'Marginal+' : 0 <= ψ < THR_LR
      'Late-rise' : ψ >= THR_LR
    """
    states = np.empty(psi.shape, dtype=object)
    states[psi <  THR_SUP] = "Suppress"
    states[(psi >= THR_SUP) & (psi < 0)]     = "Stable"
    states[(psi >= 0)       & (psi < THR_LR)] = "Marginal+"
    states[psi >= THR_LR]  = "Late-rise"
    return states


def state_fractions(states: np.ndarray) -> dict:
    """Return fraction of time spent in each ECSoC state."""
    n = len(states)
    labels = ["Suppress", "Stable", "Marginal+", "Late-rise"]
    return {lab: float(np.sum(states == lab)) / n for lab in labels}
