"""SeedForge 026: point-in-time DART quality factors combined with 025 technical families."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import month_ends
from seedforge_025_screen import evaluate_factor, expanded_factor_builders


RESULTS = Path("data/results")
DART_FILE = Path("data/dart/financials.parquet")
TECHNICAL_SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
BUILD = "026.1-20260807"

COLUMNS = {
    "ticker": "ticker", "year": "year", "receipt_date": "접수일자",
    "assets": "자산총계", "liabilities": "부채총계", "equity": "자본총계",
    "revenue": "매출액", "operating_profit": "영업이익", "net_income": "당기순이익",
    "operating_cash_flow": "영업활동현금흐름",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def load_dart_events() -> pd.DataFrame:
    if not DART_FILE.exists():
        raise FileNotFoundError(f"DART 파일이 없습니다: {DART_FILE}")
    raw = pd.read_parquet(DART_FILE)
    missing = [source for source in COLUMNS.values() if source not in raw.columns]
    if missing:
        raise ValueError(f"DART 필수 열이 없습니다: {missing}")
    frame = raw.rename(columns={source: target for target, source in COLUMNS.items()})[
        list(COLUMNS)
    ].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    frame["receipt_date"] = pd.to_datetime(
        frame["receipt_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d", errors="coerce",
    )
    for column in list(COLUMNS)[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ticker", "receipt_date", "year"]).sort_values(
        ["ticker", "year", "receipt_date"]
    )
    previous = frame.groupby("ticker", sort=False)
    frame["revenue_growth"] = safe_ratio(frame["revenue"] - previous["revenue"].shift(1), previous["revenue"].shift(1).abs())
    frame["operating_profit_growth"] = safe_ratio(
        frame["operating_profit"] - previous["operating_profit"].shift(1),
        previous["operating_profit"].shift(1).abs(),
    )
    frame["asset_growth_inverse"] = -safe_ratio(
        frame["assets"] - previous["assets"].shift(1), previous["assets"].shift(1).abs()
    )
    frame["profit_acceleration"] = frame["operating_profit_growth"] - frame.groupby(
        "ticker", sort=False
    )["operating_profit_growth"].shift(1)
    return frame


def dart_factor_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "D01_roic_assets": safe_ratio(frame["operating_profit"], frame["assets"]),
        "D02_roe": safe_ratio(frame["net_income"], frame["equity"]),
        "D03_operating_margin": safe_ratio(frame["operating_profit"], frame["revenue"]),
        "D04_net_margin": safe_ratio(frame["net_income"], frame["revenue"]),
        "D05_low_debt": -safe_ratio(frame["liabilities"], frame["equity"]),
        "D06_revenue_growth": frame["revenue_growth"],
        "D07_operating_profit_growth": frame["operating_profit_growth"],
        "D08_cashflow_quality": safe_ratio(frame["operating_cash_flow"], frame["operating_profit"]),
        "D09_cashflow_on_assets": safe_ratio(frame["operating_cash_flow"], frame["assets"]),
        "D10_accrual_quality": -safe_ratio(frame["net_income"] - frame["operating_cash_flow"], frame["assets"]),
        "D11_asset_growth_inverse": frame["asset_growth_inverse"],
        "D12_profit_acceleration": frame["profit_acceleration"],
    }


def point_in_time_matrix(
    events: pd.DataFrame,
    values: pd.Series,
    dates: pd.DatetimeIndex,
    market_columns: pd.Index,
) -> pd.DataFrame:
    work = events[["receipt_date", "ticker"]].copy()
    work["value"] = values.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["value"]).drop_duplicates(["receipt_date", "ticker"], keep="last")
    pivot = work.pivot(index="receipt_date", columns="ticker", values="value").sort_index()
    union = pivot.index.union(dates).sort_values()
    aligned = pivot.reindex(union).ffill().reindex(dates)
    ticker_map = {str(column).zfill(6): column for column in market_columns}
    usable = [ticker for ticker in aligned.columns if ticker in ticker_map]
    return aligned[usable].rename(columns=ticker_map).reindex(columns=market_columns)


def main() -> None:
    if not TECHNICAL_SCREEN.exists():
        raise FileNotFoundError("factor_screen_seedforge_025.csv가 없습니다. 025 screen을 먼저 실행하세요.")
    print(f"SeedForge 026 DART+기술 조합 (build {BUILD})")
    close, open_, volume, _eligible, _kospi = core.load_data()
    dates = pd.DatetimeIndex(month_ends(close.index))
    target = close.shift(-20) / open_.shift(-1) - 1
    events = load_dart_events()
    dart_matrices: dict[str, pd.DataFrame] = {}
    dart_rows: list[dict[str, object]] = []
    for number, (name, values) in enumerate(dart_factor_columns(events).items(), 1):
        matrix = point_in_time_matrix(events, values, dates, close.columns)
        dart_matrices[name] = matrix
        result = evaluate_factor(name, "dart", matrix, target, dates)
        dart_rows.append(result)
        print(f"[DART {number:>2}/12] {name:<28} train IC {result['train_ic']:.4f} advance {result['advance_to_combo']}")
    dart_report = pd.DataFrame(dart_rows).sort_values(
        ["advance_to_combo", "train_ic", "train_spread"], ascending=False
    )
    dart_report.to_csv(RESULTS / "factor_screen_seedforge_026_dart.csv", index=False, encoding="utf-8-sig")

    dart_selected = dart_report.loc[dart_report["advance_to_combo"]].head(3)
    technical = pd.read_csv(TECHNICAL_SCREEN)
    technical_selected = (
        technical.loc[technical["advance_to_combo"]]
        .sort_values(["group", "train_ic", "train_spread"], ascending=[True, False, False])
        .groupby("group", group_keys=False).head(1)
    )
    if dart_selected.empty or technical_selected.empty:
        raise RuntimeError("DART 또는 기술 train-only 후보가 없어 조합을 만들 수 없습니다.")

    oriented: dict[str, pd.DataFrame] = {}
    groups: dict[str, str] = {}
    for row in dart_selected.itertuples(index=False):
        oriented[row.factor] = dart_matrices[row.factor].rank(axis=1, pct=True) * int(row.direction)
        groups[row.factor] = "dart"
    technical_builders = expanded_factor_builders(close, open_, volume)
    for row in technical_selected.itertuples(index=False):
        raw = technical_builders[row.factor][1]().reindex(dates)
        oriented[row.factor] = raw.rank(axis=1, pct=True) * int(row.direction)
        groups[row.factor] = row.group

    dart_names = dart_selected["factor"].tolist()
    technical_names = technical_selected["factor"].tolist()
    combos = [
        (dart_name, *technical_combo)
        for dart_name in dart_names
        for size in (1, 2, 3)
        for technical_combo in combinations(technical_names, size)
    ]
    combo_rows: list[dict[str, object]] = []
    for combo in combos:
        composite = sum(oriented[name] for name in combo) / len(combo)
        result = evaluate_factor(" + ".join(combo), "dart_technical_combo", composite, target, dates)
        result["factor_count"] = len(combo)
        result["factor_groups"] = "+".join(groups[name] for name in combo)
        result["selected_on_train_only"] = True
        combo_rows.append(result)
    combo_report = pd.DataFrame(combo_rows).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"], ascending=False
    )
    combo_report.to_csv(RESULTS / "factor_combos_seedforge_026.csv", index=False, encoding="utf-8-sig")
    print("\ntrain-only DART+기술 상위 조합")
    columns = [
        "factor", "factor_groups", "factor_count", "train_ic", "train_ic_2014_2017",
        "train_ic_2018_2021", "train_spread", "train_selected_return",
        "opened_2022_2026_ic", "opened_2022_2026_selected_return", "advance_to_combo",
    ]
    print(combo_report[columns].head(20).to_string(index=False))
    print(f"저장: {RESULTS / 'factor_combos_seedforge_026.csv'}")
    print("주의: DART 값은 접수일 이후에만 forward-fill하며 조합 정렬은 train 값만 사용합니다.")


if __name__ == "__main__":
    main()
