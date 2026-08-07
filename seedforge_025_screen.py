"""SeedForge 025.2: expanded OHLCV factor screen selected only on 2014-2021."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import ema, month_ends


RESULTS = Path("data/results")
BUILD = "025.2-20260807"
ROUNDTRIP_COST = 0.01


def expanded_factor_builders(
    close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame
) -> dict[str, tuple[str, Callable[[], pd.DataFrame]]]:
    ret = close.pct_change(fill_method=None)
    value = close * volume
    ma = {n: close.rolling(n).mean() for n in (20, 60, 120, 200, 252)}
    std = {n: ret.rolling(n).std() for n in (20, 60, 120, 252)}
    downside = ret.clip(upper=0)
    rsi = core.rsi_frame(close)
    macd = ema(close, 12) - ema(close, 26)
    builders: dict[str, tuple[str, Callable[[], pd.DataFrame]]] = {
        "M01_ret20": ("momentum", lambda: close.pct_change(20, fill_method=None)),
        "M02_ret60": ("momentum", lambda: close.pct_change(60, fill_method=None)),
        "M03_ret120": ("momentum", lambda: close.pct_change(120, fill_method=None)),
        "M04_ret180": ("momentum", lambda: close.pct_change(180, fill_method=None)),
        "M05_ret252": ("momentum", lambda: close.pct_change(252, fill_method=None)),
        "M06_ret12m_skip1m": ("momentum", lambda: close.shift(20) / close.shift(252) - 1),
        "M07_multi_horizon": ("momentum", lambda: sum(close.pct_change(n, fill_method=None).rank(axis=1, pct=True) for n in (20, 60, 120, 252)) / 4),
        "M08_momentum_accel": ("momentum", lambda: close.pct_change(60, fill_method=None) - close.pct_change(120, fill_method=None) / 2),
        "T01_ma20_distance": ("trend", lambda: close / ma[20] - 1),
        "T02_ma60_distance": ("trend", lambda: close / ma[60] - 1),
        "T03_ma120_distance": ("trend", lambda: close / ma[120] - 1),
        "T04_ma200_distance": ("trend", lambda: close / ma[200] - 1),
        "T05_ma60_slope": ("trend", lambda: ma[60] / ma[60].shift(20) - 1),
        "T06_ma200_slope": ("trend", lambda: ma[200] / ma[200].shift(20) - 1),
        "T07_ma_alignment": ("trend", lambda: close.gt(ma[20]).astype(float) + ma[20].gt(ma[60]) + ma[60].gt(ma[120]) + ma[120].gt(ma[200])),
        "T08_ema_spread": ("trend", lambda: ema(close, 20) / ema(close, 60) - 1),
        "T09_breakout20": ("trend", lambda: close / close.rolling(20).max().shift(1) - 1),
        "T10_breakout60": ("trend", lambda: close / close.rolling(60).max().shift(1) - 1),
        "T11_near_high252": ("trend", lambda: close / close.rolling(252).max()),
        "T12_efficiency20": ("trend", lambda: close.diff(20) / close.diff().abs().rolling(20).sum()),
        "T13_efficiency60": ("trend", lambda: close.diff(60) / close.diff().abs().rolling(60).sum()),
        "R01_low_vol20": ("risk", lambda: -std[20]),
        "R02_low_vol60": ("risk", lambda: -std[60]),
        "R03_low_vol252": ("risk", lambda: -std[252]),
        "R04_downside_vol60": ("risk", lambda: -downside.rolling(60).std()),
        "R05_downside_vol252": ("risk", lambda: -downside.rolling(252).std()),
        "R06_max_loss20": ("risk", lambda: ret.rolling(20).min()),
        "R07_max_loss60": ("risk", lambda: ret.rolling(60).min()),
        "R08_drawdown252": ("risk", lambda: close / close.rolling(252).max() - 1),
        "R09_ulcer60": ("risk", lambda: -(close / close.rolling(60).max() - 1).pow(2).rolling(60).mean().pow(0.5)),
        "R10_skew60": ("risk", lambda: ret.rolling(60).skew()),
        "V01_value20": ("volume", lambda: value.rolling(20).mean()),
        "V02_value252": ("volume", lambda: value.rolling(252).mean()),
        "V03_value_growth": ("volume", lambda: value.rolling(20).mean() / value.rolling(120).mean() - 1),
        "V04_relative_volume": ("volume", lambda: volume.rolling(5).mean() / volume.rolling(60).mean() - 1),
        "V05_vwma20_spread": ("volume", lambda: close.mul(volume).rolling(20).sum() / volume.rolling(20).sum() / ma[20] - 1),
        "V06_vwma60_spread": ("volume", lambda: close.mul(volume).rolling(60).sum() / volume.rolling(60).sum() / ma[60] - 1),
        "V07_amihud20": ("volume", lambda: -(ret.abs() / value.replace(0, np.nan)).rolling(20).mean()),
        "V08_up_volume_ratio": ("volume", lambda: volume.where(ret > 0, 0).rolling(20).sum() / volume.rolling(20).sum()),
        "O01_rsi": ("oscillator", lambda: rsi - 50),
        "O02_rsi_change": ("oscillator", lambda: rsi.diff(10)),
        "O03_macd_spread": ("oscillator", lambda: macd / close),
        "O04_macd_histogram": ("oscillator", lambda: (macd - ema(macd, 9)) / close),
        "O05_roc_accel": ("oscillator", lambda: close.pct_change(20, fill_method=None) - close.pct_change(60, fill_method=None) / 3),
        "O06_reversal5": ("oscillator", lambda: -close.pct_change(5, fill_method=None)),
        "O07_reversal20": ("oscillator", lambda: -close.pct_change(20, fill_method=None)),
        "O08_coppock_proxy": ("oscillator", lambda: (close.pct_change(252, fill_method=None) + close.pct_change(120, fill_method=None)).rolling(10).mean()),
        "P01_gap_risk20": ("price_action", lambda: -(open_ / close.shift(1) - 1).rolling(20).std()),
        "P02_gap_risk60": ("price_action", lambda: -(open_ / close.shift(1) - 1).rolling(60).std()),
        "P03_positive_days20": ("price_action", lambda: ret.gt(0).rolling(20).mean()),
        "P04_positive_days60": ("price_action", lambda: ret.gt(0).rolling(60).mean()),
        "P05_close_open_strength": ("price_action", lambda: (close / open_ - 1).rolling(20).mean()),
    }
    return builders


def evaluate_factor(name: str, group: str, factor: pd.DataFrame, target: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for date in dates:
        x, y = factor.loc[date], target.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < 50:
            continue
        rank_x, rank_y = x[valid].rank(pct=True), y[valid].rank(pct=True)
        if rank_x.nunique() < 2 or rank_y.nunique() < 2:
            continue
        rows.append({
            "date": date, "ic": rank_x.corr(rank_y),
            "top": y[valid].loc[rank_x >= 0.8].mean() - ROUNDTRIP_COST,
            "bottom": y[valid].loc[rank_x <= 0.2].mean() - ROUNDTRIP_COST,
            "spread": y[valid].loc[rank_x >= 0.8].mean() - y[valid].loc[rank_x <= 0.2].mean(),
        })
    monthly = pd.DataFrame(rows)
    train = monthly.loc[monthly["date"] <= "2021-12-31"]
    early = train.loc[train["date"] <= "2017-12-31"]
    late = train.loc[train["date"] >= "2018-01-01"]
    opened = monthly.loc[monthly["date"] >= "2022-01-01"]
    direction = 1 if train["ic"].mean() >= 0 else -1
    oriented = lambda frame, column: float(frame[column].mean() * direction) if not frame.empty else float("nan")
    train_selected = float((train["top"] if direction > 0 else train["bottom"]).mean())
    opened_selected = float((opened["top"] if direction > 0 else opened["bottom"]).mean())
    train_stable = all(value > 0 for value in (
        oriented(train, "ic"), oriented(early, "ic"), oriented(late, "ic"), oriented(train, "spread")
    ))
    return {
        "factor": name, "group": group, "direction": direction,
        "train_months": len(train), "opened_months": len(opened),
        "train_ic": oriented(train, "ic"), "train_ic_2014_2017": oriented(early, "ic"),
        "train_ic_2018_2021": oriented(late, "ic"), "train_spread": oriented(train, "spread"),
        "train_selected_return": train_selected, "opened_2022_2026_ic": oriented(opened, "ic"),
        "opened_2022_2026_selected_return": opened_selected,
        "advance_to_combo": bool(train_stable and oriented(train, "ic") >= 0.01 and train_selected > 0),
    }


def main() -> None:
    print(f"SeedForge 025 확장 단독팩터 스크린 (build {BUILD})")
    close, open_, volume, _eligible, _kospi = core.load_data()
    target = close.shift(-20) / open_.shift(-1) - 1
    dates = pd.DatetimeIndex(month_ends(close.index))
    builders = expanded_factor_builders(close, open_, volume)
    rows = []
    for number, (name, (group, builder)) in enumerate(builders.items(), 1):
        result = evaluate_factor(name, group, builder(), target, dates)
        rows.append(result)
        print(f"[{number:>2}/{len(builders)}] {name:<25} train IC {result['train_ic']:.4f} advance {result['advance_to_combo']}")
    report = pd.DataFrame(rows).sort_values(["advance_to_combo", "train_ic", "train_spread"], ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS / "factor_screen_seedforge_025.csv", index=False, encoding="utf-8-sig")
    print("\n계열별 train 상위 팩터")
    print(report.groupby("group", group_keys=False).head(5).to_string(index=False))
    print(f"저장: {RESULTS / 'factor_screen_seedforge_025.csv'}")
    print("주의: 2022~2026 열은 열린 진단값이며 조합 선정에 사용하지 않습니다.")


if __name__ == "__main__":
    main()
