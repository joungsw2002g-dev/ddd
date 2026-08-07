"""SeedForge 024 phase 3: RSI-exit portfolio backtest for train-selected factor composites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import factor_builders, month_ends


RESULTS = Path("data/results")
COMBOS = RESULTS / "factor_combos_seedforge_024.csv"
SCREEN = RESULTS / "factor_screen_seedforge_024.csv"
BUILD = "024.6-20260806"


def build_signals(
    combo: tuple[str, ...], directions: dict[str, int], builders: dict[str, object],
    dates: pd.DatetimeIndex, market: core.SimulationInputs,
    market_signal_ok: np.ndarray | None = None, max_candidates: int = 80,
) -> tuple[dict[int, list[tuple[int, float]]], dict[str, int]]:
    components = []
    for name in combo:
        raw = builders[name]().reindex(dates)
        components.append(raw.rank(axis=1, pct=True) * directions[name])
    composite = sum(components) / len(components)
    signals: dict[int, list[tuple[int, float]]] = {}
    diagnostics = {"rebalance_months": 0, "blocked_months": 0, "signal_months": 0}
    positions = market.dates.get_indexer(dates)
    for row_number, day in enumerate(positions):
        if day < 0 or day + 1 >= len(market.dates):
            continue
        diagnostics["rebalance_months"] += 1
        if market_signal_ok is not None and not bool(market_signal_ok[day]):
            diagnostics["blocked_months"] += 1
            continue
        scores = composite.iloc[row_number].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(scores))
        if not valid.size:
            continue
        selected = valid[np.argsort(scores[valid])[-max_candidates:]]
        signals[int(day)] = [(int(stock), float(scores[stock])) for stock in selected]
        diagnostics["signal_months"] += 1
    return signals, diagnostics


def main() -> None:
    if not COMBOS.exists() or not SCREEN.exists():
        raise FileNotFoundError("024 screen/combo CSV가 없습니다. screen과 combo를 먼저 실행하세요.")
    combo_report = pd.read_csv(COMBOS).sort_values(["train_ic", "train_spread"], ascending=False).head(10)
    screen = pd.read_csv(SCREEN).set_index("factor")
    directions = screen["direction"].astype(int).to_dict()
    market, _features, overbought, neutral, kospi = core.load_or_prepare(False)
    close = pd.DataFrame(market.close, index=market.dates, columns=market.tickers)
    volume = pd.DataFrame(market.volume, index=market.dates, columns=market.tickers)
    builders = factor_builders(close, volume)
    dates = pd.DatetimeIndex(month_ends(market.dates))
    kospi_ok = (kospi > kospi.rolling(200).mean()).reindex(market.dates).fillna(False).to_numpy()

    summaries: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    total = len(combo_report) * 2
    number = 0
    print(f"SeedForge 024 RSI 포트폴리오 백테스트 (build {BUILD}): {total}개")
    for combo_row in combo_report.itertuples(index=False):
        combo = tuple(part.strip() for part in combo_row.factor.split("+"))
        for market_policy, market_ok in (("none", None), ("kospi_ma200", kospi_ok)):
            number += 1
            buys, signal_diagnostics = build_signals(combo, directions, builders, dates, market, market_ok)
            equity, exposure, trades, counts = core.simulate(
                market, buys, overbought, neutral, "neutral_plus_overbought", "base", None
            )
            metrics = core.performance(equity, exposure, kospi.pct_change(fill_method=None).fillna(0), trades)
            row = {
                "preset": f"combo_{combo_row.Index if hasattr(combo_row, 'Index') else number:02d}",
                "factor_combo": " + ".join(combo), "frequency": "monthly",
                "heat_policy": "factor_composite", "crash_policy": "none", "vol_policy": "none",
                "value_policy": "none", "market_policy": market_policy,
                "exit_policy": "neutral_plus_overbought", "risk_policy": "base",
                **metrics, **counts, "missed": counts["skipped"] / max(1, counts["signals"]),
                **signal_diagnostics,
            }
            row["score"] = core.score_result(row)
            summaries.append(row)
            periods.extend(core.period_rows(row, equity, exposure, kospi, trades))
            print(
                f"[{number:>2}/{total}] {market_policy:<12} CAGR {metrics['cagr']:.2%} "
                f"MDD {metrics['mdd']:.1%} PF {metrics['trade_pf']:.2f} 거래 {counts['filled']} "
                f"월 {signal_diagnostics['signal_months']}/{signal_diagnostics['rebalance_months']} "
                f"차단 {signal_diagnostics['blocked_months']}"
            )

    summary = core.add_validation_scores(pd.DataFrame(summaries), pd.DataFrame(periods)).sort_values("robust_score", ascending=False)
    summary.to_csv(RESULTS / "summary_seedforge_024_portfolio.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(RESULTS / "periods_seedforge_024_portfolio.csv", index=False, encoding="utf-8-sig")
    columns = ["factor_combo", "market_policy", "cagr", "mdd", "trade_pf", "cagr_test_2022_2026",
               "mdd_test_2022_2026", "trade_pf_test_2022_2026", "robust_score", "passes_mdd_gate"]
    print("\nRSI 포트폴리오 상위 후보")
    print(summary[columns].head(10).to_string(index=False))
    regime_rows = summary.loc[summary["market_policy"].eq("kospi_ma200")]
    if not regime_rows.empty and int(regime_rows["blocked_months"].max()) == 0:
        print("경고: KOSPI MA200 차단 월이 0개입니다. KOSPI 데이터/날짜 정렬을 확인하세요.")
    print(f"저장: {RESULTS / 'summary_seedforge_024_portfolio.csv'}")


if __name__ == "__main__":
    main()
