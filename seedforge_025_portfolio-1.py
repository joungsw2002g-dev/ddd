"""SeedForge 025.4: fresh-data RSI portfolio test for train-selected 025 combinations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import seedforge_021 as core
from seedforge_024_portfolio import add_objective_gates, build_signals
from seedforge_024_screen import month_ends
from seedforge_025_screen import expanded_factor_builders


RESULTS = Path("data/results")
COMBOS = RESULTS / "factor_combos_seedforge_025.csv"
SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
BENCHMARK = RESULTS / "benchmark_periods_seedforge_025.csv"
BENCHMARK_MONTHLY = RESULTS / "benchmark_monthly_seedforge_025.csv"
BUILD = "025.4.1-20260807-common-endpoint-r1"
TOP_COMBOS = 10


def main() -> None:
    for required in (COMBOS, SCREEN, BENCHMARK, BENCHMARK_MONTHLY):
        if not required.exists():
            raise FileNotFoundError(f"필수 결과가 없습니다: {required}")

    combo_report = pd.read_csv(COMBOS).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"],
        ascending=False,
    ).head(TOP_COMBOS)
    directions = pd.read_csv(SCREEN).set_index("factor")["direction"].astype(int).to_dict()
    benchmark = pd.read_csv(BENCHMARK)
    opened_equal_weight = benchmark.loc[
        benchmark["period"].eq("test_2022_2026"), "equal_weight_cagr"
    ]
    equal_weight_cagr = float(opened_equal_weight.iloc[0]) if not opened_equal_weight.empty else float("nan")
    benchmark_monthly = pd.read_csv(BENCHMARK_MONTHLY, parse_dates=["date"])
    last_realized_month = pd.Timestamp(benchmark_monthly["date"].max())

    # Deliberately bypass seedforge_021_prepared.pkl.  The 025 benchmark used raw
    # current files, so portfolio inputs must use the same data snapshot.
    print(f"SeedForge 025 RSI 포트폴리오 (build {BUILD}) — 원본 데이터 새로 준비")
    close, open_, volume, entry_eligible, kospi = core.load_data()
    market = core.prepare_market(close, open_, volume, entry_eligible)
    _features, overbought, neutral = core.prepare_features(close, volume, kospi)
    builders_with_groups = expanded_factor_builders(close, open_, volume)
    builders = {name: item[1] for name, item in builders_with_groups.items()}
    dates = pd.DatetimeIndex(month_ends(market.dates))
    kospi_ok = (kospi > kospi.rolling(200).mean()).reindex(market.dates).fillna(False).to_numpy()
    following_dates = market.dates[market.dates > last_realized_month]
    common_end = pd.Timestamp(following_dates[0] if len(following_dates) else market.dates[-1])
    kospi_common = kospi.loc[:common_end]
    print(f"공통 성과 종료일: {common_end.date()} (동일가중·KOSPI·전략 일치)")

    summaries: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    total = len(combo_report) * 2
    number = 0
    for combo_row in combo_report.itertuples(index=False):
        combo = tuple(part.strip() for part in combo_row.factor.split("+"))
        for market_policy, market_ok in (("none", None), ("kospi_ma200", kospi_ok)):
            number += 1
            buys, diagnostics = build_signals(combo, directions, builders, dates, market, market_ok)
            equity, exposure, trades, counts = core.simulate(
                market, buys, overbought, neutral, "neutral_plus_overbought", "base", None
            )
            equity_common = equity.loc[:common_end]
            exposure_common = exposure.reindex(equity_common.index).fillna(0)
            if trades.empty:
                trades_common = trades
            else:
                trades_common = trades.loc[pd.to_datetime(trades["exit_date"]) <= common_end].copy()
            metrics = core.performance(
                equity_common, exposure_common,
                kospi_common.pct_change(fill_method=None).reindex(equity_common.index).fillna(0),
                trades_common,
            )
            row = {
                "preset": f"025_combo_{number:02d}", "factor_combo": " + ".join(combo),
                "factor_groups": combo_row.factor_groups, "factor_count": combo_row.factor_count,
                "frequency": "monthly", "heat_policy": "factor_composite",
                "crash_policy": "none", "vol_policy": "none", "value_policy": "none",
                "market_policy": market_policy, "exit_policy": "neutral_plus_overbought",
                "risk_policy": "base", "selected_on_train_only": True,
                "opened_equal_weight_cagr_2022_2026": equal_weight_cagr,
                **metrics, **counts, **diagnostics,
            }
            row["score"] = core.score_result(row)
            summaries.append(row)
            periods.extend(core.period_rows(row, equity_common, exposure_common, kospi_common, trades_common))
            print(
                f"[{number:>2}/{total}] {market_policy:<12} CAGR {metrics['cagr']:.2%} "
                f"MDD {metrics['mdd']:.1%} PF {metrics['trade_pf']:.2f} 거래 {counts['filled']} "
                f"월 {diagnostics['signal_months']}/{diagnostics['rebalance_months']} "
                f"차단 {diagnostics['blocked_months']}"
            )

    summary = core.add_validation_scores(pd.DataFrame(summaries), pd.DataFrame(periods))
    summary = add_objective_gates(summary, kospi_common)
    summary["excess_vs_equal_weight_opened"] = (
        summary["cagr_test_2022_2026"] - summary["opened_equal_weight_cagr_2022_2026"]
    )
    summary["passes_research_hurdle_not_independent"] = summary[
        "passes_objective_gate_at_base_cost"
    ]
    summary["passes_live_gate"] = False
    summary = summary.sort_values(
        ["passes_research_hurdle_not_independent", "robust_score"], ascending=[False, False]
    )
    summary.to_csv(RESULTS / "summary_seedforge_025_portfolio.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(
        RESULTS / "periods_seedforge_025_portfolio.csv", index=False, encoding="utf-8-sig"
    )
    columns = [
        "factor_combo", "market_policy", "cagr", "mdd", "trade_pf",
        "cagr_test_2022_2026", "kospi_cagr_test_2022_2026",
        "opened_equal_weight_cagr_2022_2026", "excess_cagr_test_2022_2026",
        "excess_vs_equal_weight_opened", "mdd_test_2022_2026",
        "passes_mdd_gate", "passes_research_hurdle_not_independent", "passes_live_gate",
    ]
    print("\nSeedForge 025 RSI 포트폴리오 결과")
    print(summary[columns].head(10).to_string(index=False))
    print("\n주의: 2022~2026은 이미 열린 진단 구간이므로 어떤 결과도 실전 PASS가 아닙니다.")
    print(f"저장: {RESULTS / 'summary_seedforge_025_portfolio.csv'}")


if __name__ == "__main__":
    main()
