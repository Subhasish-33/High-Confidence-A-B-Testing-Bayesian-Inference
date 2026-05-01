"""
main.py
=======
Interactive CLI for the Bayesian A/B Testing engine.

Usage
-----
  # Single-query mode (pass data as arguments):
  python src/main.py --ca 52 --va 480 --cb 63 --vb 495

  # Interactive mode (prompts for input):
  python src/main.py

  # Run the full pipeline (generate → analyze → report):
  python src/main.py --run-full

  # Use a flat (uninformative) prior:
  python src/main.py --ca 52 --va 480 --cb 63 --vb 495 --prior flat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))
from bayesian_engine import BayesianEngine, DayResult, DEFAULT_PRIOR

# ANSI colour helpers
_R  = "\033[0m"
_B  = "\033[1m"
_TEAL   = "\033[96m"
_ORANGE = "\033[93m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

def _c(text, code): return f"{code}{text}{_R}"


def _print_banner():
    print(_c("\n╔══════════════════════════════════════════════════════════╗", _TEAL))
    print(_c("║     Bayesian A/B Testing Engine  |  Real-Time PoB CLI    ║", _TEAL))
    print(_c("╚══════════════════════════════════════════════════════════╝\n", _TEAL))


def _display_result(result: DayResult):
    """Pretty-print a single DayResult to the terminal."""
    try:
        from tabulate import tabulate

        header_rows = [
            ["Metric", "Group A (Control)", "Group B (Variant)"],
        ]
        data_rows = [
            ["Posterior Mean",
             _c(f"{result.posterior_mean_a:.4%}", _TEAL),
             _c(f"{result.posterior_mean_b:.4%}", _ORANGE)],
            ["95% HDI",
             _c(f"[{result.hdi_a[0]:.4%}, {result.hdi_a[1]:.4%}]", _TEAL),
             _c(f"[{result.hdi_b[0]:.4%}, {result.hdi_b[1]:.4%}]", _ORANGE)],
            ["Visitors", f"{result.cum_visitors_a:,}", f"{result.cum_visitors_b:,}"],
            ["Conversions", f"{result.cum_conversions_a:,}", f"{result.cum_conversions_b:,}"],
        ]
        print(tabulate(data_rows, headers=["Metric", "Group A (Control)", "Group B (Variant)"],
                       tablefmt="fancy_grid"))
    except ImportError:
        print(f"  Group A  mean={result.posterior_mean_a:.4%}  HDI=[{result.hdi_a[0]:.4%},{result.hdi_a[1]:.4%}]")
        print(f"  Group B  mean={result.posterior_mean_b:.4%}  HDI=[{result.hdi_b[0]:.4%},{result.hdi_b[1]:.4%}]")

    print()
    pob_bar_len = int(result.pob * 40)
    pob_bar = "█" * pob_bar_len + "░" * (40 - pob_bar_len)
    print(f"  {_c('Probability of Beat (B > A):', _B)} {_c(f'{result.pob:.2%}', _GREEN)}")
    print(f"  [{_c(pob_bar, _GREEN)}]")

    lift_col = _GREEN if result.relative_lift > 0 else _RED
    print(f"\n  {_c('Relative Lift:', _B)}  {_c(f'{result.relative_lift:+.3f}%', lift_col)}")
    print(f"  {_c('Expected Loss:', _B)}  {result.expected_loss:.6f}", end="")

    if result.expected_loss < 0.001:
        print(f"  {_c('← below threshold', _GREEN)}", end="")
    print()

    # Decision
    print()
    if result.should_stop:
        print(_c("  ✅  DECISION: EXPERIMENT COMPLETE — Safe to deploy Variant B.", _GREEN))
        print(_c(f"      Reason: {result.stop_reason}", _DIM))
    elif result.pob > 0.85:
        print(_c("  ⏳  DECISION: Promising — but continue collecting data.", _ORANGE))
    elif result.pob > 0.5:
        print(_c("  ⏳  DECISION: Inconclusive — continue experiment.", _DIM))
    else:
        print(_c("  ❌  DECISION: B is likely underperforming A.", _RED))
    print()


def _interactive_mode():
    """Prompt the user for data and display a result."""
    _print_banner()
    print("  Enter cumulative experiment totals:\n")
    try:
        ca = int(input("  Conversions  (Group A) : "))
        va = int(input("  Visitors     (Group A) : "))
        cb = int(input("  Conversions  (Group B) : "))
        vb = int(input("  Visitors     (Group B) : "))
    except (ValueError, EOFError):
        print(_c("\n  [Error] Please enter integer values.", _RED))
        sys.exit(1)

    prior_choice = input("\n  Prior? [informative / flat, default=informative]: ").strip().lower()
    if prior_choice == "flat":
        prior = (1, 1)
        print(_c("  Using Beta(1,1) — flat uninformative prior.\n", _DIM))
    else:
        prior = DEFAULT_PRIOR
        print(_c("  Using Beta(10,90) — informative 10% baseline prior.\n", _DIM))

    result = BayesianEngine.quick_evaluate(
        conversions_a=ca, visitors_a=va,
        conversions_b=cb, visitors_b=vb,
        prior_alpha=prior[0], prior_beta=prior[1],
    )
    _display_result(result)


def _single_query_mode(ca, va, cb, vb, prior):
    _print_banner()
    print(f"  Input  →  A: {ca}/{va} conversions   B: {cb}/{vb} conversions\n")
    result = BayesianEngine.quick_evaluate(
        conversions_a=ca, visitors_a=va,
        conversions_b=cb, visitors_b=vb,
        prior_alpha=prior[0], prior_beta=prior[1],
    )
    _display_result(result)


def _run_full_pipeline(prior):
    print(_c("\n  ── Running Full Pipeline ─────────────────────────────────────\n", _TEAL))

    # Step 1: Generate data
    print(_c("  [1/3] Generating synthetic experiment data...", _DIM))
    from generate_data import main as gen_main
    gen_main()

    # Step 2: Run analyzer
    print(_c("  [2/3] Running sequential Bayesian analysis...", _DIM))
    from analyzer import run_analysis
    from pathlib import Path
    run_analysis()

    # Step 3: Print summary
    print(_c("  [3/3] Pipeline complete. Output files:\n", _GREEN))
    out_dir = Path(__file__).parent.parent / "outputs"
    for f in sorted(out_dir.iterdir()):
        print(f"    • {f.name}  ({f.stat().st_size // 1024} KB)")
    print()

    # Show summary.json decision
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        print(_c(f"  DECISION: {data['decision']}", _GREEN))
        es = data["early_stopping"]
        if es["triggered"]:
            print(_c(f"  Early Stop: Day {es['trigger_day']} "
                     f"({es['days_saved_pct']} experiment time saved)", _GREEN))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian A/B Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ca", type=int, help="Conversions in Group A")
    parser.add_argument("--va", type=int, help="Visitors in Group A")
    parser.add_argument("--cb", type=int, help="Conversions in Group B")
    parser.add_argument("--vb", type=int, help="Visitors in Group B")
    parser.add_argument(
        "--prior", choices=["informative", "flat"], default="informative",
        help="Prior type: 'informative' = Beta(10,90), 'flat' = Beta(1,1)",
    )
    parser.add_argument(
        "--run-full", action="store_true",
        help="Run the full pipeline: generate data → analyze → report",
    )
    args = parser.parse_args()

    prior = DEFAULT_PRIOR if args.prior == "informative" else (1, 1)

    if args.run_full:
        _run_full_pipeline(prior)
    elif all(v is not None for v in [args.ca, args.va, args.cb, args.vb]):
        _single_query_mode(args.ca, args.va, args.cb, args.vb, prior)
    else:
        _interactive_mode()


if __name__ == "__main__":
    main()
