"""SeedForge 027.2: RSI portfolio test for train-selected cadence-matched DART combinations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_portfolio import add_objective_gates
from seedforge_025_screen import expanded_factor_builders
from seedforge_026 import dart_factor_columns, load_dart_events, point_in_time_matrix
from seedforge_027 import quarter_ends


RESULTS = Path("data/results")
COMBOS = RESULTS / "factor_combos_seedforge_027.csv"
SCREEN = RESULTS / "factor_screen_seedforge_027.csv"
BENCHMARK = RESULTS / "benchmark_periods_seedforge_025.csv"
BENCHMARK_MONTHLY = RESULTS / "benchmark_monthly_seedforge_025.csv"
BUILD = "027.2-20260807"
TOP_COMBOS = 10


def matrix_signals(
    composite: pd.DataFrame,
    dates: pd.DatetimeIndex,
    market: core.SimulationInputs,
    market_ok: np.ndarray | None,
    max_candidates: int = 80,
) -> tuple[dict[int, list[tuple[int, float]]], dict[str, int]]:
    signals: dict[int, list[tuple[int, float]]] = {}
    diagnostics = {"rebalance_months": 0, "blocked_months": 0, "signal_months": 0}
    positions = market.dates.get_indexer(dates)
    for row_number, day in enumerate(positions):
        if day < 0 or day + 1 >= len(market.dates):
            continue
        diagnostics["rebalance_months"] += 1
        if market_ok is not None and not bool(market_ok[day]):
            diagnostics["blocked_months"] += 1
            continue
        scores = composite.iloc[row_number].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(scores))
        if not valid.size:
            continue
        selected = valid[np.argsort(scores[valid])[-max_candidates:]]
        signals[int(day)] = [(int(column), float(scores[column])) for column in selected]
        diagnostics["signal_months"] += 1
    return signals, diagnostics


def main() -> None:
    for required in (COMBOS, SCREEN, BENCHMARK, BENCHMARK_MONTHLY):
        if not required.exists():
            raise FileNotFoundError(f"필수 결과가 없습니다: {required}")
    combo_report = pd.read_csv(COMBOS).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"], ascending=False
    ).head(TOP_COMBOS)
    factor_screen = pd.read_csv(SCREEN)
    benchmark = pd.read_csv(BENCHMARK)
    equal_weight_row = benchmark.loc[benchmark["period"].eq("test_2022_2026")]
    equal_weight_cagr = float(equal_weight_row.iloc[0]["equal_weight_cagr"])
    benchmark_monthly = pd.read_csv(BENCHMARK_MONTHLY, parse_dates=["date"])
    last_realized_month = pd.Timestamp(benchmark_monthly["date"].max())

    print(f"SeedForge 027 RSI 포트폴리오 (build {BUILD}) — 원본 데이터 새로 준비")
    close, open_, volume, entry_eligible, kospi = core.load_data()
    market = core.prepare_market(close, open_, volume, entry_eligible)
    _features, overbought, neutral = core.prepare_features(close, volume, kospi)
    dates = quarter_ends(market.dates)
    following_dates = market.dates[market.dates > last_realized_month]
    common_end = pd.Timestamp(following_dates[0] if len(following_dates) else market.dates[-1])
    kospi_common = kospi.loc[:common_end]
    kospi_ok = (kospi > kospi.rolling(200).mean()).reindex(market.dates).fillna(False).to_numpy()
    print(f"공통 성과 종료일: {common_end.date()}")

    dart_events = load_dart_events()
    dart_matrices = {
        name: point_in_time_matrix(dart_events, values, dates, close.columns)
        for name, values in dart_factor_columns(dart_events).items()
    }
    technical_builders = expanded_factor_builders(close, open_, volume)
    needed_factors = {
        part.strip() for value in combo_report["factor"] for part in value.split("+")
    }
    technical_matrices = {
        name: technical_builders[name][1]().reindex(dates)
        for name in needed_factors if name in technical_builders
    }

    summaries: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    total = len(combo_report) * 2
    number = 0
    for combo_row in combo_report.itertuples(index=False):
        names = tuple(part.strip() for part in combo_row.factor.split("+"))
        horizon = int(combo_row.horizon_days)
        direction_rows = factor_screen.loc[
            factor_screen["horizon_days"].eq(horizon)
            & factor_screen["factor"].isin(names)
        ].set_index("factor")
        missing = set(names) - set(direction_rows.index)
        if missing:
            raise ValueError(f"{horizon}일 방향 정보가 없는 팩터: {sorted(missing)}")
        components = []
        for name in names:
            matrix = dart_matrices[name] if name in dart_matrices else technical_matrices[name]
            components.append(matrix.rank(axis=1, pct=True) * int(direction_rows.loc[name, "direction"]))
        composite = sum(components) / len(components)

        for market_policy, market_ok in (("none", None), ("kospi_ma200", kospi_ok)):
            number += 1
            buys, diagnostics = matrix_signals(composite, dates, market, market_ok)
            equity, exposure, trades, counts = core.simulate(
                market, buys, overbought, neutral, "neutral_plus_overbought", "base", None
            )
            equity_common = equity.loc[:common_end]
            exposure_common = exposure.reindex(equity_common.index).fillna(0)
            trades_common = trades if trades.empty else trades.loc[
                pd.to_datetime(trades["exit_date"]) <= common_end
            ].copy()
            metrics = core.performance(
                equity_common, exposure_common,
                kospi_common.pct_change(fill_method=None).reindex(equity_common.index).fillna(0),
                trades_common,
            )
            row = {
                "preset": f"027_combo_{number:02d}", "factor_combo": " + ".join(names),
                "factor_groups": combo_row.factor_groups, "factor_count": combo_row.factor_count,
                "horizon_days": horizon, "frequency": "quarterly",
                "heat_policy": "dart_technical_composite", "crash_policy": "none",
                "vol_policy": "none", "value_policy": "none", "market_policy": market_policy,
                "exit_policy": "neutral_plus_overbought", "risk_policy": "base",
                "selected_on_train_only": True,
                "opened_equal_weight_cagr_2022_2026": equal_weight_cagr,
                **metrics, **counts, **diagnostics,
            }
            row["score"] = core.score_result(row)
            summaries.append(row)
            periods.extend(core.period_rows(row, equity_common, exposure_common, kospi_common, trades_common))
            print(
                f"[{number:>2}/{total}] {horizon:>3}일 {market_policy:<12} "
                f"CAGR {metrics['cagr']:.2%} MDD {metrics['mdd']:.1%} PF {metrics['trade_pf']:.2f}"
            )

    summary = core.add_validation_scores(pd.DataFrame(summaries), pd.DataFrame(periods))
    summary = add_objective_gates(summary, kospi_common)
    summary["excess_vs_equal_weight_opened"] = (
        summary["cagr_test_2022_2026"] - summary["opened_equal_weight_cagr_2022_2026"]
    )
    summary["passes_live_gate"] = False
    summary = summary.sort_values(["passes_objective_gate_at_base_cost", "robust_score"], ascending=False)
    summary.to_csv(RESULTS / "summary_seedforge_027_portfolio.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(RESULTS / "periods_seedforge_027_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\nSeedForge 027 RSI 포트폴리오 결과")
    columns = [
        "factor_combo", "horizon_days", "market_policy", "cagr", "mdd", "trade_pf",
        "cagr_test_2022_2026", "kospi_cagr_test_2022_2026",
        "opened_equal_weight_cagr_2022_2026", "excess_cagr_test_2022_2026",
        "excess_vs_equal_weight_opened", "mdd_test_2022_2026",
        "passes_mdd_gate", "passes_objective_gate_at_base_cost", "passes_live_gate",
    ]
    print(summary[columns].head(10).to_string(index=False))
    print("주의: 열린 2022~2026 진단이므로 실전 게이트는 항상 False입니다.")
    print(f"저장: {RESULTS / 'summary_seedforge_027_portfolio.csv'}")


if __name__ == "__main__":
    main()
