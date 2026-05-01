"""
analyzer.py — Sequential Bayesian Analyzer
Processes experiment CSV day-by-day, generates all visualizations + summary.

Outputs (in outputs/)
  posterior_evolution.gif     — animated posterior KDE over 21 days
  risk_vs_time.png            — PoB rise + Expected Loss fall, annotated
  frequentist_vs_bayesian.png — p-value vs. PoB side-by-side
  summary.json                — final metrics, HDI, decision recommendation
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).parent))
from bayesian_engine import BayesianEngine, DayResult, frequentist_p_value

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
CSV_PATH   = OUTPUT_DIR / "experiment_data.csv"

# Design tokens
DARK_BG      = "#0d1117"
DARK_SURFACE = "#161b22"
DARK_BORDER  = "#30363d"
TEAL         = "#4fc3f7"
ORANGE       = "#ff8f00"
GREEN        = "#69db7c"
RED_SOFT     = "#ff6b6b"
GRID_COLOR   = "#21262d"
TEXT_MAIN    = "#f0f6fc"
TEXT_DIM     = "#8b949e"


def _style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    axlist = [axes] if axes is not None and not hasattr(axes, "__iter__") else (axes or fig.get_axes())
    for ax in axlist:
        ax.set_facecolor(DARK_SURFACE)
        ax.tick_params(colors=TEXT_DIM, labelsize=10)
        ax.xaxis.label.set_color(TEXT_MAIN)
        ax.yaxis.label.set_color(TEXT_MAIN)
        ax.title.set_color(TEXT_MAIN)
        for s in ax.spines.values():
            s.set_edgecolor(DARK_BORDER)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)


def _posterior_frame(result: DayResult) -> bytes:
    x = np.linspace(0.01, 0.30, 500)
    pdf_a = beta_dist.pdf(x, result.alpha_a, result.beta_a)
    pdf_b = beta_dist.pdf(x, result.alpha_b, result.beta_b)

    fig, ax = plt.subplots(figsize=(9, 5))
    _style(fig, ax)

    ax.fill_between(x * 100, pdf_a, alpha=0.22, color=TEAL)
    ax.fill_between(x * 100, pdf_b, alpha=0.22, color=ORANGE)
    ax.plot(x * 100, pdf_a, color=TEAL,   lw=2.5, label=f"A Control  μ={result.posterior_mean_a:.3%}")
    ax.plot(x * 100, pdf_b, color=ORANGE, lw=2.5, label=f"B Variant  μ={result.posterior_mean_b:.3%}")
    ax.axvline(result.posterior_mean_a * 100, color=TEAL,   ls="--", lw=1.2, alpha=0.7)
    ax.axvline(result.posterior_mean_b * 100, color=ORANGE, ls="--", lw=1.2, alpha=0.7)
    ax.axvspan(result.hdi_a[0] * 100, result.hdi_a[1] * 100, alpha=0.07, color=TEAL)
    ax.axvspan(result.hdi_b[0] * 100, result.hdi_b[1] * 100, alpha=0.07, color=ORANGE)

    status_col  = GREEN if result.should_stop else TEXT_DIM
    status_txt  = "[DONE] EXPERIMENT COMPLETE" if result.should_stop else "[...] Monitoring..."
    info = (f"PoB (B>A) : {result.pob:.1%}\n"
            f"Rel Lift  : {result.relative_lift:+.2f}%\n"
            f"E[Loss]   : {result.expected_loss:.5f}\n"
            f"Status    : {status_txt}")
    ax.text(0.98, 0.97, info, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, family="monospace", color=TEXT_MAIN,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=DARK_BG,
                      edgecolor=status_col, linewidth=1.4))

    ax.set_xlabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(
        f"Posterior Distributions — Day {result.day} / 21   "
        f"({result.cum_visitors_a:,} + {result.cum_visitors_b:,} visitors)",
        fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=10)
    ax.legend(loc="upper left", framealpha=0.2, facecolor=DARK_BG,
              edgecolor=DARK_BORDER, labelcolor=TEXT_MAIN, fontsize=10)
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_posterior_gif(results: List[DayResult]) -> Path:
    frames = []
    print("  Rendering GIF frames", end="", flush=True)
    for r in results:
        img = Image.open(io.BytesIO(_posterior_frame(r))).convert("RGB")
        frames.append(img)
        print(".", end="", flush=True)
    print(" done!")

    durations = [250] * len(frames)
    for i in range(-3, 0):
        durations[i] = 900

    path = OUTPUT_DIR / "posterior_evolution.gif"
    frames[0].save(path, format="GIF", append_images=frames[1:],
                   save_all=True, duration=durations, loop=0, optimize=False)
    return path


def generate_risk_chart(results: List[DayResult]) -> Path:
    days   = [r.day for r in results]
    pobs   = [r.pob * 100 for r in results]
    losses = [r.expected_loss for r in results]
    stop   = next((r.day for r in results if r.should_stop), None)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    _style(fig, [ax1, ax2])
    ax2.set_facecolor(DARK_SURFACE)
    for s in ax2.spines.values():
        s.set_edgecolor(DARK_BORDER)
    ax2.tick_params(colors=TEXT_DIM, labelsize=10)

    ax1.plot(days, pobs, color=TEAL, lw=2.5, marker="o", ms=5, label="Prob. of Beat (%)")
    ax1.fill_between(days, pobs, alpha=0.12, color=TEAL)
    ax1.axhline(95, color=TEAL, ls="--", lw=1.2, alpha=0.6, label="95% Threshold")
    ax1.set_ylabel("Probability of Beat (%)", fontsize=11, color=TEAL)
    ax1.tick_params(axis="y", colors=TEAL)
    ax1.set_ylim(0, 105)

    ax2.plot(days, losses, color=ORANGE, lw=2.5, marker="s", ms=5, label="Expected Loss")
    ax2.fill_between(days, losses, alpha=0.12, color=ORANGE)
    ax2.axhline(0.001, color=ORANGE, ls="--", lw=1.2, alpha=0.6, label="E[Loss] Threshold")
    ax2.set_ylabel("Expected Loss", fontsize=11, color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)

    if stop:
        ax1.axvline(stop, color=GREEN, lw=2, ls=":", alpha=0.9)
        ax1.text(stop + 0.2, 20, f"  ✓ Early Stop\n  Day {stop}",
                 color=GREEN, fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                           edgecolor=GREEN, linewidth=1.2))

    ax1.set_xlabel("Experiment Day", fontsize=11)
    ax1.set_xticks(days)
    fig.suptitle("Sequential Analysis: Probability of Beat & Expected Loss Over Time",
                 fontsize=13, fontweight="bold", color=TEXT_MAIN, y=0.98)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center left", framealpha=0.2,
               facecolor=DARK_BG, edgecolor=DARK_BORDER, labelcolor=TEXT_MAIN, fontsize=10)

    plt.tight_layout(pad=2)
    out = OUTPUT_DIR / "risk_vs_time.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_comparison_chart(results: List[DayResult], df: pd.DataFrame) -> Path:
    days = [r.day for r in results]
    pobs = [r.pob * 100 for r in results]

    cum_ca = cum_va = cum_cb = cum_vb = 0
    p_values = []
    for _, row in df.iterrows():
        cum_ca += row["conversions_a"]; cum_va += row["visitors_a"]
        cum_cb += row["conversions_b"]; cum_vb += row["visitors_b"]
        p_values.append(frequentist_p_value(cum_ca, cum_va, cum_cb, cum_vb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _style(fig, [ax1, ax2])

    ax1.plot(days, p_values, color=RED_SOFT, lw=2.5, marker="o", ms=5)
    ax1.fill_between(days, p_values, alpha=0.15, color=RED_SOFT)
    ax1.axhline(0.05, color=RED_SOFT, ls="--", lw=1.5, alpha=0.8, label="p=0.05 threshold")
    ax1.set_title('Frequentist A/B Test\n(Two-Proportion Z-Test)', fontsize=12,
                  color=TEXT_MAIN, fontweight="bold")
    ax1.set_xlabel("Experiment Day", fontsize=11)
    ax1.set_ylabel("p-value (two-sided)", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(days[::2])
    ax1.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor=DARK_BORDER,
               labelcolor=TEXT_MAIN, fontsize=10)
    ax1.text(0.5, 0.88,
             '"P-Value Trap" — oscillates around\nthreshold, enables p-hacking.\nTells you IF, not HOW MUCH.',
             transform=ax1.transAxes, ha="center", va="top", fontsize=9,
             color=RED_SOFT, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                       edgecolor=RED_SOFT, lw=1, alpha=0.8))

    stop_day = next((r.day for r in results if r.should_stop), None)
    ax2.plot(days, pobs, color=TEAL, lw=2.5, marker="o", ms=5)
    ax2.fill_between(days, pobs, alpha=0.15, color=TEAL)
    ax2.axhline(95, color=GREEN, ls="--", lw=1.5, alpha=0.8, label="95% PoB threshold")
    if stop_day:
        ax2.axvline(stop_day, color=GREEN, lw=2, ls=":", alpha=0.9)
        ax2.text(stop_day + 0.2, 8, f"  Safe Early Stop\n  Day {stop_day}",
                 color=GREEN, fontsize=9.5, fontweight="bold")
    ax2.set_title('Bayesian A/B Test\n(Beta-Binomial Posterior PoB)', fontsize=12,
                  color=TEXT_MAIN, fontweight="bold")
    ax2.set_xlabel("Experiment Day", fontsize=11)
    ax2.set_ylabel("Probability of Beat (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(days[::2])
    ax2.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor=DARK_BORDER,
               labelcolor=TEXT_MAIN, fontsize=10)
    ax2.text(0.5, 0.88,
             'Monotonically converges.\nPoB = direct probability\nfor business stakeholders.',
             transform=ax2.transAxes, ha="center", va="top", fontsize=9,
             color=TEAL, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                       edgecolor=TEAL, lw=1, alpha=0.8))

    fig.suptitle("Frequentist vs. Bayesian A/B Testing — Why PoB Wins for Business Decisions",
                 fontsize=13, fontweight="bold", color=TEXT_MAIN, y=1.01)
    plt.tight_layout(pad=2)
    out = OUTPUT_DIR / "frequentist_vs_bayesian.png"
    fig.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_summary_json(results: List[DayResult]) -> Path:
    final  = results[-1]
    stop_r = next((r for r in results if r.should_stop), None)

    if final.pob > 0.95:
        decision = "DEPLOY VARIANT B — Statistical confidence achieved."
    elif final.pob > 0.5:
        decision = "CONTINUE EXPERIMENT — Insufficient evidence to declare a winner."
    else:
        decision = "ABANDON VARIANT B — Variant is likely performing worse than Control."

    payload = {
        "experiment_summary": {
            "total_days": final.day,
            "total_visitors_a": final.cum_visitors_a,
            "total_visitors_b": final.cum_visitors_b,
            "total_conversions_a": final.cum_conversions_a,
            "total_conversions_b": final.cum_conversions_b,
            "observed_rate_a": round(final.cum_conversions_a / final.cum_visitors_a, 5),
            "observed_rate_b": round(final.cum_conversions_b / final.cum_visitors_b, 5),
        },
        "bayesian_results": {
            "probability_of_beat": round(final.pob, 5),
            "probability_of_beat_pct": f"{final.pob:.2%}",
            "relative_lift_pct": round(final.relative_lift, 4),
            "expected_loss": round(final.expected_loss, 6),
            "posterior_mean_a": round(final.posterior_mean_a, 5),
            "posterior_mean_b": round(final.posterior_mean_b, 5),
            "hdi_95_a": [round(final.hdi_a[0], 5), round(final.hdi_a[1], 5)],
            "hdi_95_b": [round(final.hdi_b[0], 5), round(final.hdi_b[1], 5)],
        },
        "early_stopping": {
            "triggered": stop_r is not None,
            "trigger_day": stop_r.day if stop_r else None,
            "trigger_reason": stop_r.stop_reason if stop_r else None,
            "days_saved": (final.day - stop_r.day) if stop_r else 0,
            "days_saved_pct": f"{(final.day - stop_r.day) / final.day:.1%}" if stop_r else "0%",
        },
        "decision": decision,
        "prior_used": "Beta(10, 90) — informative 10% baseline prior",
        "n_monte_carlo_samples": 20000,
    }

    out = OUTPUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def _print_table(results: List[DayResult]):
    try:
        from tabulate import tabulate
        rows = [[
            r.day,
            f"{r.cum_visitors_a:,}", f"{r.cum_conversions_a:,}",
            f"{r.cum_visitors_b:,}", f"{r.cum_conversions_b:,}",
            f"{r.pob:.1%}", f"{r.relative_lift:+.2f}%",
            f"{r.expected_loss:.5f}",
            "✓ STOP" if r.should_stop else "",
        ] for r in results]
        headers = ["Day","Vis-A","Conv-A","Vis-B","Conv-B","PoB","Lift","E[Loss]","Decision"]
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    except ImportError:
        for r in results:
            print(f"  Day {r.day:2d}  PoB={r.pob:.1%}  Lift={r.relative_lift:+.2f}%  "
                  f"E[Loss]={r.expected_loss:.5f}  {'✓ STOP' if r.should_stop else ''}")


def run_analysis(csv_path: Path = CSV_PATH, prior: tuple = (10, 90)) -> List[DayResult]:
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("        Run  python src/generate_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    OUTPUT_DIR.mkdir(exist_ok=True)

    engine  = BayesianEngine(prior_alpha=prior[0], prior_beta=prior[1])
    results: List[DayResult] = []

    print("\n" + "=" * 68)
    print("  Bayesian Sequential Analyzer — processing experiment data")
    print("=" * 68)

    for _, row in df.iterrows():
        result = engine.update(
            day=int(row["day"]),
            visitors_a=int(row["visitors_a"]),
            conversions_a=int(row["conversions_a"]),
            visitors_b=int(row["visitors_b"]),
            conversions_b=int(row["conversions_b"]),
        )
        results.append(result)
        flag = " ← ✓ EARLY STOP TRIGGERED" if result.should_stop else ""
        print(f"  Day {result.day:2d}  PoB={result.pob:.1%}  Lift={result.relative_lift:+.2f}%  "
              f"E[Loss]={result.expected_loss:.5f}{flag}")

    print()
    _print_table(results)

    print("\n  Generating visualizations...")
    print(f"  ✓ {generate_posterior_gif(results)}")
    print(f"  ✓ {generate_risk_chart(results)}")
    print(f"  ✓ {generate_comparison_chart(results, df)}")
    print(f"  ✓ {generate_summary_json(results)}")

    final = results[-1]
    stop  = next((r for r in results if r.should_stop), None)
    print("\n" + "=" * 68)
    print("  FINAL REPORT")
    print("=" * 68)
    print(f"  Probability of Beat  : {final.pob:.2%}")
    print(f"  Relative Lift        : {final.relative_lift:+.2f}%")
    print(f"  Expected Loss        : {final.expected_loss:.6f}")
    print(f"  95% HDI for A        : [{final.hdi_a[0]:.3%}, {final.hdi_a[1]:.3%}]")
    print(f"  95% HDI for B        : [{final.hdi_b[0]:.3%}, {final.hdi_b[1]:.3%}]")
    if stop:
        saved = final.day - stop.day
        print(f"\n  Early Stop Day       : {stop.day}  (saved {saved} days = "
              f"{saved/final.day:.0%} of experiment)")
    with open(OUTPUT_DIR / "summary.json") as f:
        print(f"\n  DECISION: {json.load(f)['decision']}")
    print("=" * 68 + "\n")
    return results


if __name__ == "__main__":
    run_analysis()
