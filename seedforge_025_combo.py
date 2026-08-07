"""SeedForge 025.3: train-only cross-family combinations from the 52-factor screen."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_024_screen import month_ends
from seedforge_025_screen import evaluate_factor, expanded_factor_builders


RESULTS = Path("data/results")
SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
OUTPUT = RESULTS / "factor_combos_seedforge_025.csv"
BUILD = "025.3-20260807"
MAX_PER_GROUP = 2
MAX_CORRELATION = 0.85


def mean_rank_correlation(left: pd.DataFrame, right: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    correlations: list[float] = []
    for date in dates:
        x, y = left.loc[date], right.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < 50:
            continue
        value = x[valid].rank(pct=True).corr(y[valid].rank(pct=True))
        if np.isfinite(value):
            correlations.append(abs(float(value)))
    return float(np.mean(correlations)) if correlations else float("nan")


def remove_correlated(
    candidates: pd.DataFrame,
    oriented: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    kept: list[str] = []
    audit: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        blocker, correlation = "", 0.0
        for existing in kept:
            value = mean_rank_correlation(oriented[row.factor], oriented[existing], train_dates)
            if np.isfinite(value) and value >= MAX_CORRELATION:
                blocker, correlation = existing, value
                break
        keep = not blocker
        if keep:
            kept.append(row.factor)
        audit.append({
            "factor": row.factor, "group": row.group, "kept": keep,
            "blocked_by": blocker, "mean_abs_rank_correlation": correlation,
        })
    return candidates.loc[candidates["factor"].isin(kept)].copy(), audit


def main() -> None:
    if not SCREEN.exists():
        raise FileNotFoundError("factor_screen_seedforge_025.csv가 없습니다. seedforge_025_screen.py를 먼저 실행하세요.")
    screen = pd.read_csv(SCREEN)
    eligible = screen.loc[screen["advance_to_combo"].eq(True)].copy()  # noqa: E712
    eligible = (
        eligible.sort_values(["group", "train_ic", "train_spread"], ascending=[True, False, False])
        .groupby("group", group_keys=False)
        .head(MAX_PER_GROUP)
        .sort_values(["train_ic", "train_spread"], ascending=False)
    )
    if len(eligible) < 2:
        raise RuntimeError("train-only 조합 후보가 2개 미만입니다.")

    close, open_, volume, _entry, _kospi = core.load_data()
    target = close.shift(-20) / open_.shift(-1) - 1
    dates = pd.DatetimeIndex(month_ends(close.index))
    train_dates = dates[dates <= "2021-12-31"]
    builders = expanded_factor_builders(close, open_, volume)
    oriented: dict[str, pd.DataFrame] = {}
    for row in eligible.itertuples(index=False):
        raw = builders[row.factor][1]().reindex(dates)
        oriented[row.factor] = raw.rank(axis=1, pct=True) * int(row.direction)

    eligible, correlation_audit = remove_correlated(eligible, oriented, train_dates)
    names = eligible["factor"].tolist()
    groups = eligible.set_index("factor")["group"].to_dict()
    combos = [
        combo
        for size in (2, 3, 4)
        for combo in combinations(names, size)
        if len({groups[name] for name in combo}) == size
    ]
    if not combos:
        raise RuntimeError("상관·계열 중복 제거 후 생성 가능한 조합이 없습니다.")

    print(
        f"SeedForge 025 계열분산 조합 스크린 (build {BUILD}): "
        f"후보 {len(names)}개, 조합 {len(combos)}개"
    )
    rows: list[dict[str, object]] = []
    for number, combo in enumerate(combos, 1):
        composite = sum(oriented[name] for name in combo) / len(combo)
        result = evaluate_factor(" + ".join(combo), "cross_family_combo", composite, target, dates)
        result["factor_count"] = len(combo)
        result["factor_groups"] = "+".join(groups[name] for name in combo)
        result["selected_on_train_only"] = True
        rows.append(result)
        if number % 25 == 0 or number == len(combos):
            print(
                f"[{number:>3}/{len(combos)}] train IC {result['train_ic']:.4f} "
                f"train 선택수익 {result['train_selected_return']:.3%}"
            )

    report = pd.DataFrame(rows).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"],
        ascending=False,
    )
    report.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(correlation_audit).to_csv(
        RESULTS / "factor_correlation_audit_seedforge_025.csv", index=False, encoding="utf-8-sig"
    )
    print("\ntrain-only 상위 계열분산 조합")
    columns = [
        "factor", "factor_groups", "factor_count", "train_ic", "train_ic_2014_2017",
        "train_ic_2018_2021", "train_spread", "train_selected_return",
        "opened_2022_2026_ic", "opened_2022_2026_selected_return", "advance_to_combo",
    ]
    print(report[columns].head(20).to_string(index=False))
    print(f"저장: {OUTPUT}")
    print("주의: 정렬과 조합 선택은 train 열만 사용했습니다. opened 열은 진단 전용입니다.")


if __name__ == "__main__":
    main()
