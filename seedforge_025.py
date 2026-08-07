"""SeedForge 025 phase 0: decompose KOSPI versus the tradable stock universe.

This diagnostic deliberately runs before another factor search.  It determines
whether the 024 failure came from stock selection or from cap-weighted index
concentration, and freezes benchmark series for later 025 combinations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import month_ends


RESULTS = Path("data/results")
BUILD = "025.1-20260807"
START = "2014-01-01"


def compound(values: pd.Series) -> float:
    clean = values.dropna().clip(lower=-0.999999)
    return float((1 + clean).prod() - 1) if not clean.empty else float("nan")


def annualized(values: pd.Series) -> float:
    clean = values.dropna().clip(lower=-0.999999)
    return float((1 + clean).prod() ** (12 / len(clean)) - 1) if not clean.empty else float("nan")


def max_drawdown(values: pd.Series) -> float:
    equity = (1 + values.fillna(0).clip(lower=-0.999999)).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else float("nan")


def build_monthly_diagnostics(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    eligible: pd.DataFrame,
    kospi: pd.Series,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(month_ends(close.loc[START:].index))
    next_open = open_.shift(-1).reindex(dates)
    stock_returns = next_open.shift(-1) / next_open - 1
    tradable = eligible.reindex(dates).fillna(False) & next_open.notna() & next_open.shift(-1).notna()
    selected = stock_returns.where(tradable)
    equal_weight = selected.mean(axis=1)
    median_stock = selected.median(axis=1)
    breadth = selected.gt(0).sum(axis=1) / selected.notna().sum(axis=1).replace(0, np.nan)
    positive = selected.clip(lower=0)
    top2 = np.sort(positive.fillna(0).to_numpy(), axis=1)[:, -2:].sum(axis=1)
    positive_sum = positive.sum(axis=1).replace(0, np.nan)
    top2_positive_share = pd.Series(top2, index=dates) / positive_sum
    kospi_monthly = kospi.reindex(dates).shift(-1) / kospi.reindex(dates) - 1
    report = pd.DataFrame({
        "date": dates,
        "kospi_return": kospi_monthly.to_numpy(),
        "equal_weight_return": equal_weight.to_numpy(),
        "median_stock_return": median_stock.to_numpy(),
        "positive_stock_ratio": breadth.to_numpy(),
        "top2_positive_contribution_share": top2_positive_share.to_numpy(),
        "tradable_stocks": selected.notna().sum(axis=1).to_numpy(),
    })
    report["kospi_minus_equal_weight"] = report["kospi_return"] - report["equal_weight_return"]
    return report.dropna(subset=["kospi_return", "equal_weight_return"])


def yearly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = [(str(year), group) for year, group in monthly.groupby(monthly["date"].dt.year)]
    periods.extend([
        ("train_2014_2021", monthly.loc[monthly["date"] <= "2021-12-31"]),
        ("test_2022_2026", monthly.loc[monthly["date"] >= "2022-01-01"]),
        ("full", monthly),
    ])
    for period, frame in periods:
        if frame.empty:
            continue
        rows.append({
            "period": period,
            "months": len(frame),
            "kospi_return": compound(frame["kospi_return"]),
            "equal_weight_return": compound(frame["equal_weight_return"]),
            "kospi_cagr": annualized(frame["kospi_return"]),
            "equal_weight_cagr": annualized(frame["equal_weight_return"]),
            "kospi_mdd": max_drawdown(frame["kospi_return"]),
            "equal_weight_mdd": max_drawdown(frame["equal_weight_return"]),
            "kospi_minus_equal_weight_cagr": annualized(frame["kospi_return"]) - annualized(frame["equal_weight_return"]),
            "median_positive_stock_ratio": frame["positive_stock_ratio"].median(),
            "median_top2_positive_share": frame["top2_positive_contribution_share"].median(),
            "avg_tradable_stocks": frame["tradable_stocks"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"SeedForge 025 벤치마크 분해 (build {BUILD})")
    close, open_, _volume, eligible, kospi = core.load_data()
    monthly = build_monthly_diagnostics(close, open_, eligible, kospi)
    yearly = yearly_summary(monthly)
    RESULTS.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(RESULTS / "benchmark_monthly_seedforge_025.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(RESULTS / "benchmark_periods_seedforge_025.csv", index=False, encoding="utf-8-sig")
    print("\n기간별 KOSPI vs 거래가능 종목 동일가중")
    print(yearly.tail(5).to_string(index=False))
    test = yearly.loc[yearly["period"].eq("test_2022_2026")]
    if not test.empty:
        gap = float(test.iloc[0]["kospi_minus_equal_weight_cagr"])
        print(f"\n2022~2026 KOSPI-동일가중 CAGR 격차: {gap:.2%}p")
    print(f"저장: {RESULTS / 'benchmark_monthly_seedforge_025.csv'}")
    print(f"저장: {RESULTS / 'benchmark_periods_seedforge_025.csv'}")
    print("다음 단계: 이 두 CSV를 확인한 뒤 025 확장 팩터 조합의 목표를 KOSPI/동일가중/잔차수익으로 분리합니다.")


if __name__ == "__main__":
    main()
