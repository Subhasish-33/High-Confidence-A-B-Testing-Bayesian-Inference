"""
generate_data.py
================
Simulates a 21-day live A/B experiment and writes a daily-snapshot CSV.

Groups
------
  A (Control) : true conversion rate λ_A = 10.5%
  B (Variant) : true conversion rate λ_B = 12.0%

Daily traffic per group is Poisson(μ=500), clamped to [400, 600].
The seed is fixed for reproducibility.

Output
------
  outputs/experiment_data.csv
    Columns: day, visitors_a, conversions_a, visitors_b, conversions_b
"""

import os
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
SEED          = 42
N_DAYS        = 21
LAMBDA_A      = 0.105   # true conversion rate for Control
LAMBDA_B      = 0.120   # true conversion rate for Variant
TRAFFIC_MU    = 500     # Poisson mean for daily visitors
TRAFFIC_MIN   = 400
TRAFFIC_MAX   = 600
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "experiment_data.csv")


def generate_experiment(
    n_days: int = N_DAYS,
    lambda_a: float = LAMBDA_A,
    lambda_b: float = LAMBDA_B,
    traffic_mu: int = TRAFFIC_MU,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic A/B experiment dataset.

    Returns a DataFrame with one row per day containing visitor and
    conversion counts for both groups.
    """
    rng = np.random.default_rng(seed)

    records = []
    for day in range(1, n_days + 1):
        # Daily traffic: Poisson, clamped to realistic bounds
        visitors_a = int(np.clip(rng.poisson(traffic_mu), TRAFFIC_MIN, TRAFFIC_MAX))
        visitors_b = int(np.clip(rng.poisson(traffic_mu), TRAFFIC_MIN, TRAFFIC_MAX))

        # Binomial draws: each visitor either converts or not
        conversions_a = int(rng.binomial(visitors_a, lambda_a))
        conversions_b = int(rng.binomial(visitors_b, lambda_b))

        records.append({
            "day":           day,
            "visitors_a":    visitors_a,
            "conversions_a": conversions_a,
            "visitors_b":    visitors_b,
            "conversions_b": conversions_b,
        })

    return pd.DataFrame(records)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_experiment()

    df.to_csv(OUTPUT_FILE, index=False)

    # ── Console summary ────────────────────────────────────────────────────────
    total_a = df["visitors_a"].sum()
    total_b = df["visitors_b"].sum()
    conv_a  = df["conversions_a"].sum()
    conv_b  = df["conversions_b"].sum()

    print("=" * 60)
    print("  Synthetic A/B Experiment — Data Generation Complete")
    print("=" * 60)
    print(f"  Output       : {OUTPUT_FILE}")
    print(f"  Days         : {len(df)}")
    print(f"  Total visitors A : {total_a:,}   (true λ = {LAMBDA_A:.1%})")
    print(f"  Total visitors B : {total_b:,}   (true λ = {LAMBDA_B:.1%})")
    print(f"  Observed rate A  : {conv_a/total_a:.3%}")
    print(f"  Observed rate B  : {conv_b/total_b:.3%}")
    print("=" * 60)
    print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
