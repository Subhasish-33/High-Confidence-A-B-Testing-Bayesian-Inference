"""
bayesian_engine.py
==================
Beta-Binomial conjugate prior Bayesian A/B Testing engine.

Mathematical Framework
----------------------
  Prior  : Beta(α₀, β₀)
  Update : α_post = α₀ + Σconversions
           β_post = β₀ + Σ(visitors - conversions)
  Posterior draws : 20,000 Monte Carlo samples per group

Key Metrics
-----------
  PoB (Probability of Beat) : P(θ_B > θ_A)
  Relative Lift             : (μ_B - μ_A) / μ_A × 100
  Expected Loss             : E[max(θ_A - θ_B, 0)]  ← regret of choosing B
  95% HDI                   : scipy.stats.beta credible interval

Stopping Rule
-------------
  If Expected Loss < 0.001 AND PoB > 0.95 → signal "EXPERIMENT COMPLETE"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy import stats


# ── Configuration ──────────────────────────────────────────────────────────────
N_SAMPLES          = 20_000
STOP_LOSS_THRESH   = 0.001   # E[Loss] threshold for early stopping
STOP_POB_THRESH    = 0.95    # PoB threshold for early stopping
DEFAULT_PRIOR      = (10, 90)  # Beta(10, 90) — informative 10% baseline prior


# ── Result dataclass ────────────────────────────────────────────────────────────
@dataclass
class DayResult:
    day:              int
    # Cumulative totals
    cum_visitors_a:   int
    cum_conversions_a: int
    cum_visitors_b:   int
    cum_conversions_b: int
    # Posterior parameters
    alpha_a:          float
    beta_a:           float
    alpha_b:          float
    beta_b:           float
    # Metrics
    pob:              float   # Probability of Beat (B > A)
    relative_lift:    float   # (μ_B - μ_A) / μ_A × 100
    expected_loss:    float   # E[max(θ_A - θ_B, 0)]
    hdi_a:            Tuple[float, float]  # 95% HDI for group A
    hdi_b:            Tuple[float, float]  # 95% HDI for group B
    posterior_mean_a: float
    posterior_mean_b: float
    should_stop:      bool = False
    stop_reason:      str  = ""
    # Monte Carlo samples (for plotting) — not serialized to JSON
    samples_a:        np.ndarray = field(default=None, repr=False, compare=False)
    samples_b:        np.ndarray = field(default=None, repr=False, compare=False)


# ── Engine ──────────────────────────────────────────────────────────────────────
class BayesianEngine:
    """
    Stateful Bayesian engine that accumulates experiment data and updates
    Beta posteriors incrementally.

    Parameters
    ----------
    prior_alpha, prior_beta : float
        Hyperparameters for the shared Beta prior.  Default Beta(10, 90)
        encodes a 10% historical baseline over 100 pseudo-observations.
    n_samples : int
        Number of Monte Carlo draws per update call.
    rng_seed : int | None
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        prior_alpha: float = DEFAULT_PRIOR[0],
        prior_beta:  float = DEFAULT_PRIOR[1],
        n_samples:   int   = N_SAMPLES,
        rng_seed:    int | None = 42,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta  = prior_beta
        self.n_samples   = n_samples
        self.rng         = np.random.default_rng(rng_seed)

        # Cumulative totals (reset on construction)
        self._cum_visitors_a    = 0
        self._cum_conversions_a = 0
        self._cum_visitors_b    = 0
        self._cum_conversions_b = 0

    # ── Public API ───────────────────────────────────────────────────────────────
    def update(
        self,
        day:           int,
        visitors_a:    int,
        conversions_a: int,
        visitors_b:    int,
        conversions_b: int,
    ) -> DayResult:
        """
        Ingest one day's data, update posteriors, and return a DayResult.
        """
        # Accumulate totals
        self._cum_visitors_a    += visitors_a
        self._cum_conversions_a += conversions_a
        self._cum_visitors_b    += visitors_b
        self._cum_conversions_b += conversions_b

        # Posterior parameters (closed-form Beta-Binomial update)
        alpha_a = self.prior_alpha + self._cum_conversions_a
        beta_a  = self.prior_beta  + (self._cum_visitors_a - self._cum_conversions_a)
        alpha_b = self.prior_alpha + self._cum_conversions_b
        beta_b  = self.prior_beta  + (self._cum_visitors_b - self._cum_conversions_b)

        # Monte Carlo simulation
        samples_a = self.rng.beta(alpha_a, beta_a, size=self.n_samples)
        samples_b = self.rng.beta(alpha_b, beta_b, size=self.n_samples)

        # ── Metrics ──────────────────────────────────────────────────────────────
        pob           = float(np.mean(samples_b > samples_a))
        mean_a        = float(samples_a.mean())
        mean_b        = float(samples_b.mean())
        relative_lift = float((mean_b - mean_a) / mean_a * 100)

        # Expected Loss: average regret of deploying B when A might be better
        expected_loss = float(np.mean(np.maximum(samples_a - samples_b, 0)))

        # 95% Highest Density Interval (equal-tail approximation via PPF)
        hdi_a = (
            float(stats.beta.ppf(0.025, alpha_a, beta_a)),
            float(stats.beta.ppf(0.975, alpha_a, beta_a)),
        )
        hdi_b = (
            float(stats.beta.ppf(0.025, alpha_b, beta_b)),
            float(stats.beta.ppf(0.975, alpha_b, beta_b)),
        )

        # ── Stopping rule ─────────────────────────────────────────────────────────
        should_stop = False
        stop_reason = ""
        if expected_loss < STOP_LOSS_THRESH and pob > STOP_POB_THRESH:
            should_stop = True
            stop_reason = (
                f"E[Loss]={expected_loss:.5f} < {STOP_LOSS_THRESH} "
                f"AND PoB={pob:.1%} > {STOP_POB_THRESH:.0%}"
            )

        return DayResult(
            day=day,
            cum_visitors_a=self._cum_visitors_a,
            cum_conversions_a=self._cum_conversions_a,
            cum_visitors_b=self._cum_visitors_b,
            cum_conversions_b=self._cum_conversions_b,
            alpha_a=alpha_a,
            beta_a=beta_a,
            alpha_b=alpha_b,
            beta_b=beta_b,
            pob=pob,
            relative_lift=relative_lift,
            expected_loss=expected_loss,
            hdi_a=hdi_a,
            hdi_b=hdi_b,
            posterior_mean_a=mean_a,
            posterior_mean_b=mean_b,
            should_stop=should_stop,
            stop_reason=stop_reason,
            samples_a=samples_a,
            samples_b=samples_b,
        )

    def reset(self) -> None:
        """Reset cumulative counters (useful for repeated runs)."""
        self._cum_visitors_a    = 0
        self._cum_conversions_a = 0
        self._cum_visitors_b    = 0
        self._cum_conversions_b = 0

    # ── Static helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def quick_evaluate(
        conversions_a: int,
        visitors_a:    int,
        conversions_b: int,
        visitors_b:    int,
        prior_alpha:   float = DEFAULT_PRIOR[0],
        prior_beta:    float = DEFAULT_PRIOR[1],
        n_samples:     int   = N_SAMPLES,
        rng_seed:      int | None = 42,
    ) -> DayResult:
        """
        One-shot evaluation given aggregated totals (no state).
        Used by the CLI for single-query mode.
        """
        engine = BayesianEngine(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            n_samples=n_samples,
            rng_seed=rng_seed,
        )
        return engine.update(
            day=1,
            visitors_a=visitors_a,
            conversions_a=conversions_a,
            visitors_b=visitors_b,
            conversions_b=conversions_b,
        )


# ── Frequentist helper (for comparison chart) ─────────────────────────────────
def frequentist_p_value(
    conversions_a: int,
    visitors_a:    int,
    conversions_b: int,
    visitors_b:    int,
) -> float:
    """
    Two-proportion z-test p-value (two-sided).
    Returns 1.0 if insufficient data.
    """
    if visitors_a == 0 or visitors_b == 0:
        return 1.0

    p_a  = conversions_a / visitors_a
    p_b  = conversions_b / visitors_b
    p_pool = (conversions_a + conversions_b) / (visitors_a + visitors_b)

    denom = np.sqrt(p_pool * (1 - p_pool) * (1 / visitors_a + 1 / visitors_b))
    if denom == 0:
        return 1.0

    z = (p_b - p_a) / denom
    p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    return p_val
