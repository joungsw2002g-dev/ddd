"""SeedForge 028: fixed market-leadership switch between stock alpha, KOSPI, and cash."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_portfolio import build_signals
from seedforge_024_screen import month_ends
from seedforge_025_screen import expanded_factor_builders


RESULTS = Path("data/results")
COMBOS = RESULTS / "factor_combos_seedforge_025.csv"
SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
BENCHMARK_MONTHLY = RESULTS / "benchmark_monthly_seedforge_025.csv"
BUILD = "028.1-20260807"
LEADERSHIP_GAP = 0.10
BREADTH_FLOOR = 0.50
SWITCH_COST = 0.001


def metrics(returns: pd.Series) -> tuple[float, float]:
    clean = returns.dropna().clip(lower=-0.999999)
    if clean.empty:
        return float("nan"), float("nan")
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    equity = (1 + clean).cumprod()
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    mdd = float((equity / equity.cummax() - 1).min())
    return cagr, mdd


def main() -> None:
    for required in (COMBOS, SCREEN, BENCHMARK_MONTHLY):
        if not required.exists():
            raise FileNotFoundError(f"필수 결과가 없습니다: {required}")
    combo = pd.read_csv(COMBOS).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"], ascending=False
    ).iloc[0]
    names = tuple(part.strip() for part in combo.factor.split("+"))
    directions = pd.read_csv(SCREEN).set_index("factor")["direction"].astype(int).to_dict()
    benchmark_monthly = pd.read_csv(BENCHMARK_MONTHLY, parse_dates=["date"])
    last_realized_month = pd.Timestamp(benchmark_monthly["date"].max())

    print(f"SeedForge 028 시장리더십 전환 (build {BUILD}) — 원본 데이터 새로 준비")
    close, open_, volume, entry_eligible, kospi = core.load_data()
    market = core.prepare_market(close, open_, volume, entry_eligible)
    _features, overbought, neutral = core.prepare_features(close, volume, kospi)
    dates = pd.DatetimeIndex(month_ends(market.dates))
    builders = {name: item[1] for name, item in expanded_factor_builders(close, open_, volume).items()}
    buys, _diagnostics = build_signals(names, directions, builders, dates, market, None)
    stock_equity, _stock_exposure, _trades, _counts = core.simulate(
        market, buys, overbought, neutral, "neutral_plus_overbought", "base", None
    )
    following_dates = market.dates[market.dates > last_realized_month]
    common_end = pd.Timestamp(following_dates[0] if len(following_dates) else market.dates[-1])
    index = market.dates[market.dates <= common_end]
    stock_return = stock_equity.reindex(index).pct_change(fill_method=None).fillna(0)
    kospi_return = kospi.reindex(index).pct_change(fill_method=None).fillna(0)

    stock_daily = close.reindex(index).pct_change(fill_method=None)
    eligible_yesterday = entry_eligible.reindex(index).shift(1).fillna(False)
    equal_weight_return = stock_daily.where(eligible_yesterday).mean(axis=1).fillna(0)
    equal_weight_index = (1 + equal_weight_return).cumprod()
    kospi_120 = kospi.reindex(index).pct_change(120, fill_method=None)
    equal_weight_120 = equal_weight_index.pct_change(120, fill_method=None)
    breadth = (
        close.reindex(index).gt(close.reindex(index).rolling(200).mean()) & eligible_yesterday
    ).sum(axis=1) / eligible_yesterday.sum(axis=1).replace(0, np.nan)
    kospi_above_ma200 = kospi.reindex(index) > kospi.reindex(index).rolling(200).mean()
    concentrated = (
        kospi_above_ma200
        & ((kospi_120 - equal_weight_120) > LEADERSHIP_GAP)
        & (breadth < BREADTH_FLOOR)
    )
    regime = pd.Series("cash", index=index, dtype="object")
    regime.loc[kospi_above_ma200] = "stock"
    regime.loc[concentrated] = "kospi"
    regime = regime.shift(1).fillna("cash")

    hybrid_return = pd.Series(0.0, index=index)
    hybrid_return.loc[regime.eq("stock")] = stock_return.loc[regime.eq("stock")]
    hybrid_return.loc[regime.eq("kospi")] = kospi_return.loc[regime.eq("kospi")]
    hybrid_return.loc[regime.ne(regime.shift(1))] -= SWITCH_COST

    rows: list[dict[str, object]] = []
    for period, start in (("train_2014_2021", "2014-01-01"), ("opened_2022_2026", "2022-01-01"), ("full", str(index[0].date()))):
        end = "2021-12-31" if period == "train_2014_2021" else str(common_end.date())
        window = slice(start, end)
        hybrid_cagr, hybrid_mdd = metrics(hybrid_return.loc[window])
        stock_cagr, stock_mdd = metrics(stock_return.loc[window])
        kospi_cagr, kospi_mdd = metrics(kospi_return.loc[window])
        regime_window = regime.loc[window]
        rows.append({
            "period": period, "start": start, "end": end,
            "factor_combo": " + ".join(names),
            "hybrid_cagr": hybrid_cagr, "hybrid_mdd": hybrid_mdd,
            "stock_cagr": stock_cagr, "stock_mdd": stock_mdd,
            "kospi_cagr": kospi_cagr, "kospi_mdd": kospi_mdd,
            "hybrid_excess_vs_kospi": hybrid_cagr - kospi_cagr,
            "stock_days": int(regime_window.eq("stock").sum()),
            "kospi_days": int(regime_window.eq("kospi").sum()),
            "cash_days": int(regime_window.eq("cash").sum()),
            "switches": int(regime_window.ne(regime_window.shift(1)).sum() - 1),
            "passes_live_gate": False,
        })
    report = pd.DataFrame(rows)
    report.to_csv(RESULTS / "summary_seedforge_028.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "date": index, "regime": regime, "hybrid_return": hybrid_return,
        "stock_return": stock_return, "kospi_return": kospi_return,
        "breadth": breadth, "leadership_gap_120": kospi_120 - equal_weight_120,
    }).to_csv(RESULTS / "regimes_seedforge_028.csv", index=False, encoding="utf-8-sig")
    print("\nSeedForge 028 고정 시장전환 결과")
    print(report.to_string(index=False))
    print("주의: 임계값은 고정 진단 규칙이며 열린 2022~2026 결과는 실전 승인이 아닙니다.")
    print(f"저장: {RESULTS / 'summary_seedforge_028.csv'}")


if __name__ == "__main__":
    main()
