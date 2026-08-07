"""SeedForge 024 phase 2: combine non-duplicate predictive OHLCV factors."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import evaluate, factor_builders, month_ends


RESULTS = Path("data/results")
SCREEN = RESULTS / "factor_screen_seedforge_024.csv"
BUILD = "024.4-20260806"
DUPLICATE_GROUPS = (
    {"A4_low_volatility", "G21_atr_proxy_percentile"},
    {"A5_trend_strength", "G3_ma200_distance"},
    {"G7_efficiency_ratio", "G28_efficiency_ratio"},
)


def remove_known_duplicates(screen: pd.DataFrame) -> pd.DataFrame:
    keep = set(screen["factor"])
    for group in DUPLICATE_GROUPS:
        members = screen.loc[screen["factor"].isin(group)].sort_values("train_ic", ascending=False)
        keep -= set(members["factor"].iloc[1:])
    return screen.loc[screen["factor"].isin(keep)]


def main() -> None:
    if not SCREEN.exists():
        raise FileNotFoundError("factor_screen_seedforge_024.csv가 없습니다. 먼저 seedforge_024_screen.py를 실행하세요.")
    screen = pd.read_csv(SCREEN)
    eligible = screen.loc[(screen["train_ic"] > 0.01) & (screen["train_spread"] > 0)]
    eligible = remove_known_duplicates(eligible).sort_values(["train_ic", "train_spread"], ascending=False).head(8)
    if len(eligible) < 2:
        raise RuntimeError("조합할 예측 안정 팩터가 2개 미만입니다.")

    close, open_, volume, _entry, _kospi = core.load_data()
    target = close.shift(-20) / open_.shift(-1) - 1
    dates = pd.DatetimeIndex(month_ends(close.index))
    builders = factor_builders(close, volume)
    oriented: dict[str, pd.DataFrame] = {}
    for row in eligible.itertuples(index=False):
        raw = builders[row.factor]().reindex(dates)
        oriented[row.factor] = raw.rank(axis=1, pct=True) * int(row.direction)

    rows: list[dict[str, object]] = []
    candidate_names = list(oriented)
    combos = [*combinations(candidate_names, 2), *combinations(candidate_names, 3)]
    print(f"SeedForge 024 조합 스크리닝 (build {BUILD}): {len(candidate_names)}개 대표 팩터, {len(combos)}개 조합")
    for number, combo in enumerate(combos, 1):
        composite = sum(oriented[name] for name in combo) / len(combo)
        result = evaluate(" + ".join(combo), composite, target, dates)
        result["factor_count"] = len(combo)
        rows.append(result)
        if number % 10 == 0 or number == len(combos):
            print(f"[{number:>3}/{len(combos)}] {result['factor']:<75} test IC {result['test_ic']:.4f}")
    report = pd.DataFrame(rows).sort_values(["train_ic", "train_spread"], ascending=False)
    report["selected_on_train_only"] = True
    report.to_csv(RESULTS / "factor_combos_seedforge_024.csv", index=False, encoding="utf-8-sig")
    print("\n상위 팩터 조합")
    print(report.head(20).to_string(index=False))
    print(f"저장: {RESULTS / 'factor_combos_seedforge_024.csv'}")
    print("주의: 20일 스크린 결과이며, 다음 단계에서 RSI 매도법을 포함한 포트폴리오 백테스트가 필요합니다.")


if __name__ == "__main__":
    main()
