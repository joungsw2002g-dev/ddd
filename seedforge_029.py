"""SeedForge 029: point-in-time monthly investor-flow factor screen.

Factor direction and promotion are selected only on 2014-2021.  Results from
2022 onward are emitted as opened diagnostics and never enter the gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core


BUILD = "029.0-20260807"
FLOW_CANDIDATES = (
    Path("data/flow/flow_monthly.parquet"),
    Path("data/flows/flow_monthly.parquet"),
)
RESULTS = Path("data/results")
OUTPUT = RESULTS / "factor_screen_seedforge_029.csv"
HORIZONS = (20, 60)
GATE_HORIZON = 20
ROUNDTRIP_COST = 0.01
TRAIN_END = pd.Timestamp("2021-12-31")
OPENED_START = pd.Timestamp("2022-01-01")


def find_flow_file() -> Path:
    for path in FLOW_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in FLOW_CANDIDATES)
    raise FileNotFoundError(f"월별 수급 parquet가 없습니다. 확인 위치: {searched}")


def load_flow(path: Path, tickers: pd.Index) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path, columns=["date", "investor", "ticker", "netbuy"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["investor"] = frame["investor"].astype(str).str.strip()
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    required_investors = {"외국인", "기관합계", "개인"}
    missing_investors = required_investors.difference(frame["investor"].unique())
    if missing_investors:
        raise ValueError(f"필수 투자자 구분이 없습니다: {sorted(missing_investors)}")
    if frame[["date", "investor", "ticker", "netbuy"]].isna().any().any():
        raise ValueError("수급 데이터의 필수 컬럼에 결측값이 있습니다.")
    duplicates = int(frame.duplicated(["date", "investor", "ticker"]).sum())
    if duplicates:
        raise ValueError(f"date+investor+ticker 중복이 {duplicates:,}건 있습니다.")

    matrices: dict[str, pd.DataFrame] = {}
    normalized_tickers = pd.Index([str(ticker).zfill(6) for ticker in tickers])
    if normalized_tickers.has_duplicates:
        raise ValueError("OHLCV 종목코드를 6자리 문자열로 변환한 뒤 중복이 발생했습니다.")
    for investor in sorted(required_investors):
        subset = frame.loc[frame["investor"].eq(investor)]
        matrix = (
            subset.pivot(index="date", columns="ticker", values="netbuy")
            .sort_index()
            .reindex(columns=normalized_tickers)
            .astype(float)
        )
        matrix.columns = tickers
        matrices[investor] = matrix
    common_dates = matrices["외국인"].index
    for matrix in matrices.values():
        common_dates = common_dates.union(matrix.index)
    return {name: matrix.reindex(common_dates) for name, matrix in matrices.items()}


def asof_daily_matrix(daily: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return the last trading-day observation known by each month-end."""
    positions = daily.index.searchsorted(signal_dates, side="right") - 1
    result = pd.DataFrame(np.nan, index=signal_dates, columns=daily.columns)
    valid = positions >= 0
    if valid.any():
        result.iloc[np.flatnonzero(valid)] = daily.iloc[positions[valid]].to_numpy()
    return result


def forward_target(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    entry_eligible: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """Buy strictly after month-end and sell after ``horizon`` sessions."""
    entry_positions = close.index.searchsorted(signal_dates, side="right")
    exit_positions = entry_positions + horizon - 1
    target = pd.DataFrame(np.nan, index=signal_dates, columns=close.columns)
    valid = exit_positions < len(close.index)
    if valid.any():
        entries = open_.iloc[entry_positions[valid]].to_numpy(dtype=float)
        exits = close.iloc[exit_positions[valid]].to_numpy(dtype=float)
        eligible = entry_eligible.iloc[entry_positions[valid]].to_numpy(dtype=bool)
        values = np.where(eligible, exits / entries - 1, np.nan)
        target.iloc[np.flatnonzero(valid)] = values
    return target


def build_factors(
    flows: dict[str, pd.DataFrame], close: pd.DataFrame, volume: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    foreign = flows["외국인"]
    institution = flows["기관합계"]
    individual = flows["개인"]
    dates = foreign.index
    combined = foreign + institution
    foreign_rank = foreign.rank(axis=1, pct=True)
    institution_rank = institution.rank(axis=1, pct=True)
    co_buy = foreign_rank + institution_rank
    persistence = (
        foreign.gt(0).rolling(3, min_periods=3).sum()
        + institution.gt(0).rolling(3, min_periods=3).sum()
    ).where(foreign.rolling(3).count().eq(3) & institution.rolling(3).count().eq(3))
    acceleration = combined - combined.shift(1).rolling(3, min_periods=3).mean()

    # Monthly net buying is divided by a trailing daily traded-value baseline
    # known at the signal month-end. Cross-sectional ranks cap its influence.
    traded_value = (close * volume).rolling(20, min_periods=10).mean()
    known_value = asof_daily_matrix(traded_value, dates).replace(0, np.nan)
    normalized = combined / known_value

    return {
        "F01_foreign_netbuy_1m": foreign,
        "F02_institution_netbuy_1m": institution,
        "F03_individual_inverse_1m": -individual,
        "F04_foreign_institution_1m": combined,
        "F05_foreign_institution_cobuy": co_buy,
        "F06_foreign_netbuy_3m": foreign.rolling(3, min_periods=3).sum(),
        "F07_institution_netbuy_3m": institution.rolling(3, min_periods=3).sum(),
        "F08_foreign_institution_3m": combined.rolling(3, min_periods=3).sum(),
        "F09_foreign_netbuy_6m": foreign.rolling(6, min_periods=6).sum(),
        "F10_flow_persistence_3m": persistence,
        "F11_flow_acceleration": acceleration,
        "F12_flow_to_traded_value": normalized,
    }


def horizon_statistics(
    factor: pd.DataFrame, target: pd.DataFrame, direction: int
) -> dict[str, float | int]:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date in factor.index.intersection(target.index):
        x, y = factor.loc[date], target.loc[date]
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 50:
            continue
        ranks = x[valid].rank(pct=True)
        target_ranks = y[valid].rank(pct=True)
        if ranks.nunique() < 2 or target_ranks.nunique() < 2:
            continue
        top = y[valid].loc[ranks >= 0.8]
        bottom = y[valid].loc[ranks <= 0.2]
        rows.append({
            "date": date,
            "ic": float(ranks.corr(target_ranks)),
            "spread": float(top.mean() - bottom.mean()),
            "selected_return": float((top if direction > 0 else bottom).mean() - ROUNDTRIP_COST),
            "observations": int(valid.sum()),
        })
    monthly = pd.DataFrame(rows)

    def summarize(frame: pd.DataFrame) -> tuple[int, float, float, float, float]:
        if frame.empty:
            return 0, np.nan, np.nan, np.nan, np.nan
        return (
            len(frame),
            float(frame["ic"].mean() * direction),
            float(frame["spread"].mean() * direction),
            float(frame["selected_return"].mean()),
            float(frame["observations"].median()),
        )

    train = monthly.loc[monthly["date"] <= TRAIN_END]
    early = train.loc[train["date"] <= "2017-12-31"]
    late = train.loc[train["date"] >= "2018-01-01"]
    opened = monthly.loc[monthly["date"] >= OPENED_START]
    train_n, train_ic, train_spread, train_return, median_n = summarize(train)
    early_n, early_ic, _early_spread, _early_return, _ = summarize(early)
    late_n, late_ic, _late_spread, _late_return, _ = summarize(late)
    opened_n, opened_ic, opened_spread, opened_return, _ = summarize(opened)
    return {
        "train_periods": train_n,
        "train_ic": train_ic,
        "train_ic_2014_2017": early_ic,
        "train_ic_2018_2021": late_ic,
        "train_spread": train_spread,
        "train_selected_return_net": train_return,
        "train_median_observations": median_n,
        "early_periods": early_n,
        "late_periods": late_n,
        "opened_periods": opened_n,
        "opened_ic": opened_ic,
        "opened_spread": opened_spread,
        "opened_selected_return_net": opened_return,
    }


def evaluate_factor(
    name: str, factor: pd.DataFrame, targets: dict[int, pd.DataFrame]
) -> dict[str, object]:
    gate_target = targets[GATE_HORIZON]
    raw_train_ics: list[float] = []
    for date in factor.index.intersection(gate_target.index):
        if date > TRAIN_END:
            continue
        valid = factor.loc[date].notna() & gate_target.loc[date].notna()
        if int(valid.sum()) >= 50:
            raw_train_ics.append(
                float(factor.loc[date, valid].rank(pct=True).corr(gate_target.loc[date, valid].rank(pct=True)))
            )
    if not raw_train_ics:
        raise ValueError(f"{name}: 방향을 정할 수 있는 train 관측치가 없습니다.")
    direction = 1 if np.nanmean(raw_train_ics) >= 0 else -1
    row: dict[str, object] = {
        "factor": name,
        "direction": direction,
        "gate_horizon_days": GATE_HORIZON,
        "selected_on_train_only": True,
    }
    stats_by_horizon: dict[int, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        stats = horizon_statistics(factor, targets[horizon], direction)
        stats_by_horizon[horizon] = stats
        row.update({f"{key}_{horizon}d": value for key, value in stats.items()})
    gate = stats_by_horizon[GATE_HORIZON]
    enough_history = int(gate["early_periods"]) >= 24 and int(gate["late_periods"]) >= 36
    row["advance_to_combo"] = bool(
        enough_history
        and float(gate["train_ic"]) >= 0.01
        and float(gate["train_ic_2014_2017"]) > 0
        and float(gate["train_ic_2018_2021"]) > 0
        and float(gate["train_spread"]) > 0
        and float(gate["train_selected_return_net"]) > 0
    )
    return row


def main() -> None:
    flow_path = find_flow_file()
    print(f"SeedForge 029 수급 train-only 스크린 (build {BUILD})")
    print(f"수급 파일: {flow_path}")
    close, open_, volume, entry_eligible, _kospi = core.load_data()
    flows = load_flow(flow_path, close.columns)
    signal_dates = pd.DatetimeIndex(flows["외국인"].index)
    targets = {
        horizon: forward_target(close, open_, entry_eligible, signal_dates, horizon)
        for horizon in HORIZONS
    }
    factors = build_factors(flows, close, volume)
    rows: list[dict[str, object]] = []
    for number, (name, factor) in enumerate(factors.items(), 1):
        row = evaluate_factor(name, factor, targets)
        rows.append(row)
        print(
            f"[{number:>2}/{len(factors)}] {name:<34} "
            f"train IC {row['train_ic_20d']:.4f} "
            f"net {row['train_selected_return_net_20d']:.4f} "
            f"advance {row['advance_to_combo']}"
        )
    report = pd.DataFrame(rows).sort_values(
        ["advance_to_combo", "train_ic_20d", "train_spread_20d", "train_selected_return_net_20d"],
        ascending=False,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    display = [
        "factor", "direction", "train_periods_20d", "train_ic_20d",
        "train_ic_2014_2017_20d", "train_ic_2018_2021_20d", "train_spread_20d",
        "train_selected_return_net_20d", "opened_ic_20d",
        "opened_selected_return_net_20d", "train_ic_60d", "advance_to_combo",
    ]
    print("\n수급 단독 train-only 결과")
    print(report[display].to_string(index=False))
    print(f"\n저장: {OUTPUT}")
    print("주의: 정렬·방향·진출은 2014~2021의 20일 결과만 사용했습니다.")
    print("2022년 이후 열은 이미 열린 진단값이며 선택이나 실전 승인에 사용하지 않습니다.")


if __name__ == "__main__":
    main()
