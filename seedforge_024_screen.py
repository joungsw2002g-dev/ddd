"""SeedForge 024 phase 1: point-in-time single-factor screen on available OHLCV."""

from __future__ import annotations

import errno
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core


RESULTS = Path("data/results")
COST = 0.01
BUILD = "024.4-20260806"


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index, index=index).groupby(index.to_period("M")).last().array


def ema(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    return frame.ewm(span=span, adjust=False, min_periods=span).mean()


def factor_builders(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, Callable[[], pd.DataFrame]]:
    returns = close.pct_change(fill_method=None)
    value = close * volume
    ma20, ma60, ma120, ma200 = (close.rolling(n).mean() for n in (20, 60, 120, 200))
    vol252 = returns.rolling(252).std()
    return {
        "A1_momentum12m": lambda: close.pct_change(252, fill_method=None),
        "A2_momentum3m": lambda: close.pct_change(60, fill_method=None),
        "A3_reversal1m": lambda: -close.pct_change(20, fill_method=None),
        "A4_low_volatility": lambda: -vol252,
        "A5_trend_strength": lambda: close / ma200 - 1,
        "A6_value_growth": lambda: value.rolling(20).mean() / value.rolling(120).mean() - 1,
        "A7_near_high": lambda: close / close.rolling(252).max(),
        "A8_liquidity": lambda: value.rolling(252).mean(),
        "G1_ma_slope": lambda: (ma20 / ma20.shift(20) + ma60 / ma60.shift(20) + ma120 / ma120.shift(20) + ma200 / ma200.shift(20)) / 4 - 1,
        "G2_ma_alignment": lambda: (close.gt(ma20).astype(float) + ma20.gt(ma60) + ma60.gt(ma120) + ma120.gt(ma200)),
        "G3_ma200_distance": lambda: close / ma200 - 1,
        "G5_rising_ma200": lambda: ma200 / ma200.shift(20) - 1,
        "G7_efficiency_ratio": lambda: close.diff(20) / close.diff().abs().rolling(20).sum(),
        "G9_vwma_spread": lambda: (close.mul(volume).rolling(20).sum() / volume.rolling(20).sum()) / ma20 - 1,
        "G11_macd_acceleration": lambda: (ema(close, 12) - ema(close, 26) - ema(ema(close, 12) - ema(close, 26), 9)).diff(5),
        "G13_roc_acceleration": lambda: close.pct_change(20, fill_method=None) - close.pct_change(60, fill_method=None) / 3,
        "G14_rsi_regime": lambda: core.rsi_frame(close) - 50,
        "G21_atr_proxy_percentile": lambda: -returns.abs().rolling(20).mean().rank(axis=1, pct=True),
        "G22_bandwidth": lambda: -(4 * close.rolling(20).std() / ma20),
        "G28_efficiency_ratio": lambda: close.diff(60) / close.diff().abs().rolling(60).sum(),
        "G29_relative_volume": lambda: volume.rolling(5).mean() / volume.rolling(60).mean() - 1,
        "G35_breakout_strength": lambda: (close / close.rolling(60).max().shift(1) - 1) / returns.rolling(20).std(),
    }


def evaluate(name: str, factor: pd.DataFrame, target: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, object]:
    monthly: list[dict[str, float | pd.Timestamp]] = []
    for date in dates:
        x, y = factor.loc[date], target.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < 50:
            continue
        xv, yv = x[valid], y[valid]
        ranks = xv.rank(pct=True)
        target_ranks = yv.rank(pct=True)
        if ranks.nunique() < 2 or target_ranks.nunique() < 2:
            continue
        top = yv.loc[ranks >= 0.8]
        bottom = yv.loc[ranks <= 0.2]
        monthly.append({
            "date": date, "ic": ranks.corr(target_ranks),
            "top_return": top.mean() - COST, "bottom_return": bottom.mean() - COST,
            "spread": top.mean() - bottom.mean(),
        })
    result = pd.DataFrame(monthly)
    train = result.loc[result["date"] <= "2021-12-31"]
    test = result.loc[result["date"] >= "2022-01-01"]
    direction = 1 if train["ic"].mean() >= 0 else -1
    train_ic = train["ic"].mean() * direction
    test_ic = test["ic"].mean() * direction
    train_spread = train["spread"].mean() * direction
    test_spread = test["spread"].mean() * direction
    train_selected = (train["top_return"] if direction > 0 else train["bottom_return"]).mean()
    test_selected = (test["top_return"] if direction > 0 else test["bottom_return"]).mean()
    predictive_stable = train_ic > 0.01 and test_ic > 0 and train_spread > 0 and test_spread > 0
    cost_viable = train_selected > 0 and test_selected > 0
    return {
        "factor": name, "direction": direction, "train_months": len(train), "test_months": len(test),
        "train_ic": train_ic, "test_ic": test_ic,
        "train_spread": train_spread, "test_spread": test_spread,
        "train_top_return": train_selected, "test_top_return": test_selected,
        "predictive_stable": bool(predictive_stable),
        "cost_viable_20d": bool(cost_viable),
        "advance_to_combo": bool(predictive_stable),
        "stable": bool(predictive_stable),
    }


def main() -> None:
    close, _open, volume, _eligible, _kospi = core.load_data()
    target = close.shift(-20) / _open.shift(-1) - 1
    dates = pd.DatetimeIndex(month_ends(close.index))
    rows = []
    builders = factor_builders(close, volume)
    print(f"SeedForge 024 단독 팩터 스크리닝 (build {BUILD}): {len(builders)}개")
    for number, (name, builder) in enumerate(builders.items(), 1):
        row = evaluate(name, builder(), target, dates)
        rows.append(row)
        print(f"[{number:>2}/{len(builders)}] {name:<28} train IC {row['train_ic']:.4f} test IC {row['test_ic']:.4f} stable {row['stable']}")
    report = pd.DataFrame(rows).sort_values(["stable", "test_ic", "train_ic"], ascending=False)
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        report.to_csv(RESULTS / "factor_screen_seedforge_024.csv", index=False, encoding="utf-8-sig")
        print(f"저장: {RESULTS / 'factor_screen_seedforge_024.csv'}")
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        print("디스크 공간 부족으로 CSV 저장은 생략합니다.")
    print("\n상위 단독 팩터")
    print(report.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
