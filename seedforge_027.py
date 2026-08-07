"""SeedForge 027: cadence-matched DART and technical combinations at 3/6/12 months."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_025_screen import expanded_factor_builders
from seedforge_026 import dart_factor_columns, load_dart_events, point_in_time_matrix


RESULTS = Path("data/results")
TECHNICAL_SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
BUILD = "027.1-20260807"
COST = 0.01
HORIZONS = (60, 120, 252)


def quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period("Q")).last().array)


def evaluate(
    name: str,
    group: str,
    factor: pd.DataFrame,
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizon: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for date in dates:
        x, y = factor.loc[date], target.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < 50:
            continue
        rank_x, rank_y = x[valid].rank(pct=True), y[valid].rank(pct=True)
        if rank_x.nunique() < 2 or rank_y.nunique() < 2:
            continue
        top = y[valid].loc[rank_x >= 0.8]
        bottom = y[valid].loc[rank_x <= 0.2]
        rows.append({
            "date": date, "ic": rank_x.corr(rank_y),
            "top": top.mean() - COST, "bottom": bottom.mean() - COST,
            "spread": top.mean() - bottom.mean(),
        })
    monthly = pd.DataFrame(rows)
    train = monthly.loc[monthly["date"] <= "2021-12-31"]
    early = train.loc[train["date"] <= "2017-12-31"]
    late = train.loc[train["date"] >= "2018-01-01"]
    opened = monthly.loc[monthly["date"] >= "2022-01-01"]
    direction = 1 if train["ic"].mean() >= 0 else -1

    def oriented(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].mean() * direction) if not frame.empty else float("nan")

    train_selected = float((train["top"] if direction > 0 else train["bottom"]).mean())
    opened_selected = float((opened["top"] if direction > 0 else opened["bottom"]).mean())
    enough_history = len(early) >= 4 and len(late) >= 8
    stable = enough_history and all(value > 0 for value in (
        oriented(train, "ic"), oriented(early, "ic"), oriented(late, "ic"), oriented(train, "spread")
    ))
    return {
        "factor": name, "group": group, "horizon_days": horizon, "direction": direction,
        "train_periods": len(train), "opened_periods": len(opened),
        "train_ic": oriented(train, "ic"), "train_ic_2014_2017": oriented(early, "ic"),
        "train_ic_2018_2021": oriented(late, "ic"), "train_spread": oriented(train, "spread"),
        "train_selected_return": train_selected, "opened_2022_2026_ic": oriented(opened, "ic"),
        "opened_2022_2026_selected_return": opened_selected,
        "advance_to_combo": bool(stable and oriented(train, "ic") >= 0.01 and train_selected > 0),
    }


def main() -> None:
    if not TECHNICAL_SCREEN.exists():
        raise FileNotFoundError("factor_screen_seedforge_025.csv가 없습니다.")
    print(f"SeedForge 027 DART 보유주기 정합 조합 (build {BUILD})")
    close, open_, volume, _eligible, _kospi = core.load_data()
    dates = quarter_ends(close.index)
    events = load_dart_events()
    dart_matrices = {
        name: point_in_time_matrix(events, values, dates, close.columns)
        for name, values in dart_factor_columns(events).items()
    }
    technical_screen = pd.read_csv(TECHNICAL_SCREEN)
    technical_pool = (
        technical_screen.loc[technical_screen["advance_to_combo"]]
        .sort_values(["group", "train_ic", "train_spread"], ascending=[True, False, False])
        .groupby("group", group_keys=False).head(2)
    )
    technical_builders = expanded_factor_builders(close, open_, volume)
    technical_matrices = {
        row.factor: technical_builders[row.factor][1]().reindex(dates)
        for row in technical_pool.itertuples(index=False)
    }

    single_rows: list[dict[str, object]] = []
    combo_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        target = close.shift(-horizon) / open_.shift(-1) - 1
        dart_results = [
            evaluate(name, "dart", matrix, target, dates, horizon)
            for name, matrix in dart_matrices.items()
        ]
        technical_results = [
            evaluate(name, str(technical_pool.set_index("factor").loc[name, "group"]), matrix, target, dates, horizon)
            for name, matrix in technical_matrices.items()
        ]
        single_rows.extend(dart_results)
        single_rows.extend(technical_results)
        dart_selected = pd.DataFrame(dart_results).loc[lambda x: x["advance_to_combo"]].sort_values(
            ["train_ic", "train_spread"], ascending=False
        ).head(2)
        technical_selected = (
            pd.DataFrame(technical_results).loc[lambda x: x["advance_to_combo"]]
            .sort_values(["group", "train_ic", "train_spread"], ascending=[True, False, False])
            .groupby("group", group_keys=False).head(1)
        )
        if dart_selected.empty or technical_selected.empty:
            print(f"[{horizon:>3}일] 조합 생략: 통과 DART {len(dart_selected)}, 기술 {len(technical_selected)}")
            continue
        technical_names = technical_selected["factor"].tolist()
        technical_groups = technical_selected.set_index("factor")["group"].to_dict()
        for dart_row in dart_selected.itertuples(index=False):
            oriented_dart = dart_matrices[dart_row.factor].rank(axis=1, pct=True) * int(dart_row.direction)
            for size in (1, 2):
                for technical_combo in combinations(technical_names, size):
                    components = [oriented_dart]
                    for name in technical_combo:
                        row = technical_selected.loc[technical_selected["factor"].eq(name)].iloc[0]
                        components.append(technical_matrices[name].rank(axis=1, pct=True) * int(row["direction"]))
                    combo = (dart_row.factor, *technical_combo)
                    result = evaluate(" + ".join(combo), "dart_technical_combo", sum(components) / len(components), target, dates, horizon)
                    result["factor_count"] = len(combo)
                    result["factor_groups"] = "dart+" + "+".join(technical_groups[name] for name in technical_combo)
                    result["selected_on_train_only"] = True
                    combo_rows.append(result)

    single_report = pd.DataFrame(single_rows).sort_values(
        ["advance_to_combo", "train_ic", "train_spread"], ascending=False
    )
    single_report.to_csv(RESULTS / "factor_screen_seedforge_027.csv", index=False, encoding="utf-8-sig")
    if not combo_rows:
        print("\n모든 보유주기에서 사전등록 통과 조합이 없습니다. 027 FAIL.")
        return
    combo_report = pd.DataFrame(combo_rows).sort_values(
        ["advance_to_combo", "train_ic", "train_spread", "train_selected_return"], ascending=False
    )
    combo_report.to_csv(RESULTS / "factor_combos_seedforge_027.csv", index=False, encoding="utf-8-sig")
    print("\ntrain-only DART 보유주기 정합 상위 조합")
    columns = [
        "factor", "factor_groups", "horizon_days", "factor_count", "train_ic",
        "train_ic_2014_2017", "train_ic_2018_2021", "train_spread", "train_selected_return",
        "opened_2022_2026_ic", "opened_2022_2026_selected_return", "advance_to_combo",
    ]
    print(combo_report[columns].head(20).to_string(index=False))
    print(f"저장: {RESULTS / 'factor_combos_seedforge_027.csv'}")
    print("주의: 분기 리밸런싱·3/6/12개월 보유이며 1회 왕복비용 1%를 차감했습니다.")


if __name__ == "__main__":
    main()
