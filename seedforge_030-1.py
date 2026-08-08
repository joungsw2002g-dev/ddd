"""SeedForge 030: fixed-regime diagnostics for the SeedForge 029 flow finalist.

This is a research diagnostic, not an independent validation.  The concentrated
leadership rule is reused unchanged from SeedForge 028; no dates or stock names
are manually excluded after observing 2022-2026.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_025_screen import expanded_factor_builders
from seedforge_028 import BREADTH_FLOOR, LEADERSHIP_GAP
from seedforge_029 import (
    ROUNDTRIP_COST,
    build_factors,
    find_flow_file,
    forward_target,
    load_flow,
)


BUILD = "030.1-20260808"
RESULTS = Path("data/results")
FLOW_SCREEN = RESULTS / "factor_screen_seedforge_029.csv"
TECHNICAL_SCREEN = RESULTS / "factor_screen_seedforge_025.csv"
DETAIL_OUTPUT = RESULTS / "regime_monthly_seedforge_030.csv"
SUMMARY_OUTPUT = RESULTS / "summary_seedforge_030.csv"
FLOW_FACTOR = "F12_flow_to_traded_value"
TRAIN_END = pd.Timestamp("2021-12-31")
OPENED_START = pd.Timestamp("2022-01-01")
HORIZON = 20
NW_LAGS = 3
FAMILYWISE_ALPHA = 0.05

# Fixed before this run.  Opened results never add or remove combinations.
COMBINATIONS = (
    (FLOW_FACTOR,),
    (FLOW_FACTOR, "V01_value20"),
    (FLOW_FACTOR, "M07_multi_horizon"),
    (FLOW_FACTOR, "T02_ma60_distance"),
    (FLOW_FACTOR, "R10_skew60"),
    (FLOW_FACTOR, "O08_coppock_proxy"),
    (FLOW_FACTOR, "V01_value20", "M07_multi_horizon"),
    (FLOW_FACTOR, "V01_value20", "R10_skew60"),
    (FLOW_FACTOR, "T02_ma60_distance", "R10_skew60"),
)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def require_train_selected_factors() -> dict[str, int]:
    for path in (FLOW_SCREEN, TECHNICAL_SCREEN):
        if not path.exists():
            raise FileNotFoundError(f"필수 train-only 결과가 없습니다: {path}")
    flow = pd.read_csv(FLOW_SCREEN).set_index("factor")
    technical = pd.read_csv(TECHNICAL_SCREEN).set_index("factor")
    if FLOW_FACTOR not in flow.index or not is_true(flow.loc[FLOW_FACTOR, "advance_to_combo"]):
        raise RuntimeError(f"{FLOW_FACTOR}가 029 train-only 게이트를 통과하지 않았습니다.")
    directions = {FLOW_FACTOR: int(flow.loc[FLOW_FACTOR, "direction"])}
    required_technical = {name for combo in COMBINATIONS for name in combo if name != FLOW_FACTOR}
    for name in sorted(required_technical):
        if name not in technical.index:
            raise RuntimeError(f"025 결과에 사전등록 기술 팩터가 없습니다: {name}")
        if not is_true(technical.loc[name, "advance_to_combo"]):
            raise RuntimeError(f"025 train-only 미통과 팩터를 030에 사용할 수 없습니다: {name}")
        directions[name] = int(technical.loc[name, "direction"])
    return directions


def asof_daily_matrix(daily: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Select last-known rows without coercing mixed boolean/numeric columns."""
    if not daily.index.is_monotonic_increasing:
        daily = daily.sort_index()
    if daily.index.has_duplicates:
        raise ValueError("as-of 대상 일별 데이터의 날짜가 중복되었습니다.")
    union = daily.index.union(signal_dates).sort_values()
    return daily.reindex(union).ffill().reindex(signal_dates)


def fixed_regimes(
    close: pd.DataFrame,
    entry_eligible: pd.DataFrame,
    kospi: pd.Series,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Classify each signal using only information available at month-end."""
    stock_return = close.pct_change(fill_method=None)
    eligible_yesterday = entry_eligible.shift(1).fillna(False)
    equal_weight_return = stock_return.where(eligible_yesterday).mean(axis=1).fillna(0)
    equal_weight_index = (1 + equal_weight_return).cumprod()
    kospi_120 = kospi.pct_change(120, fill_method=None)
    equal_weight_120 = equal_weight_index.pct_change(120, fill_method=None)
    breadth = (
        close.gt(close.rolling(200).mean()) & eligible_yesterday
    ).sum(axis=1) / eligible_yesterday.sum(axis=1).replace(0, np.nan)
    kospi_above_ma200 = kospi > kospi.rolling(200).mean()
    leadership_gap = kospi_120 - equal_weight_120
    concentrated = (
        kospi_above_ma200
        & leadership_gap.gt(LEADERSHIP_GAP)
        & breadth.lt(BREADTH_FLOOR)
    )
    daily = pd.DataFrame({
        "kospi_above_ma200": kospi_above_ma200,
        "leadership_gap_120": leadership_gap,
        "breadth": breadth,
        "concentrated": concentrated,
    })
    known = asof_daily_matrix(daily, signal_dates)
    regime = pd.Series("unavailable", index=signal_dates, dtype="object")
    ready = known[["kospi_above_ma200", "leadership_gap_120", "breadth"]].notna().all(axis=1)
    regime.loc[ready] = "defensive"
    regime.loc[ready & known["kospi_above_ma200"].eq(1)] = "broad_bull"
    regime.loc[ready & known["concentrated"].eq(1)] = "concentrated"
    known["regime"] = regime
    return known


def kospi_forward_proxy(
    kospi: pd.Series, signal_dates: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    """Next-close benchmark proxy because the core KOSPI input has no open."""
    entry_positions = kospi.index.searchsorted(signal_dates, side="right")
    exit_positions = entry_positions + horizon - 1
    result = pd.Series(np.nan, index=signal_dates, dtype=float)
    valid = exit_positions < len(kospi)
    if valid.any():
        result.iloc[np.flatnonzero(valid)] = (
            kospi.iloc[exit_positions[valid]].to_numpy(dtype=float)
            / kospi.iloc[entry_positions[valid]].to_numpy(dtype=float)
            - 1
        )
    return result


def oriented_rank(frame: pd.DataFrame, direction: int) -> pd.DataFrame:
    rank = frame.rank(axis=1, pct=True)
    return rank if direction > 0 else 1 - rank


def monthly_combo_rows(
    name: str,
    components: tuple[pd.DataFrame, ...],
    target: pd.DataFrame,
    kospi_target: pd.Series,
    regimes: pd.DataFrame,
) -> list[dict[str, object]]:
    score = sum(components) / len(components)
    rows: list[dict[str, object]] = []
    for date in score.index.intersection(target.index):
        x, y = score.loc[date], target.loc[date]
        valid = x.notna() & y.notna()
        if (
            int(valid.sum()) < 50
            or pd.isna(kospi_target.loc[date])
            or regimes.loc[date, "regime"] == "unavailable"
        ):
            continue
        ranks = x[valid].rank(pct=True)
        selected = y[valid].loc[ranks >= 0.8]
        if selected.empty:
            continue
        rows.append({
            "date": date,
            "factor_combo": name,
            "factor_count": len(components),
            "sample": "train" if date <= TRAIN_END else "opened",
            "regime": regimes.loc[date, "regime"],
            "strategy_return_net": float(selected.mean() - ROUNDTRIP_COST),
            "equal_weight_return_gross": float(y[valid].mean()),
            "kospi_return_next_close_proxy": float(kospi_target.loc[date]),
            "rank_ic": float(ranks.corr(y[valid].rank(pct=True))),
            "selected_stocks": int(len(selected)),
            "tradable_stocks": int(valid.sum()),
            "leadership_gap_120": float(regimes.loc[date, "leadership_gap_120"]),
            "breadth": float(regimes.loc[date, "breadth"]),
        })
    return rows


def compounded_metrics(returns: pd.Series) -> tuple[float, float]:
    clean = returns.dropna().clip(lower=-0.999999)
    if clean.empty:
        return np.nan, np.nan
    equity = (1 + clean).cumprod()
    cagr = float(equity.iloc[-1] ** (12 / len(clean)) - 1)
    mdd = float((equity / equity.cummax() - 1).min())
    return cagr, mdd


def newey_west_t_stat(values: pd.Series, lags: int = NW_LAGS) -> float:
    """HAC t-statistic for a monthly mean, without a scipy dependency."""
    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < lags + 3:
        return np.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned / n)
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = float(demeaned[lag:] @ demeaned[:-lag] / n)
        long_run_variance += 2 * (1 - lag / (lags + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / n)
    return float(clean.mean() / standard_error) if standard_error > 0 else np.nan


def summarize_group(
    frame: pd.DataFrame, combo: str, sample: str, regime_group: str
) -> dict[str, object]:
    strategy_cagr, strategy_mdd = compounded_metrics(frame["strategy_return_net"])
    kospi_cagr, kospi_mdd = compounded_metrics(frame["kospi_return_next_close_proxy"])
    equal_weight_cagr, equal_weight_mdd = compounded_metrics(frame["equal_weight_return_gross"])
    excess_kospi = frame["strategy_return_net"] - frame["kospi_return_next_close_proxy"]
    excess_equal_weight = frame["strategy_return_net"] - frame["equal_weight_return_gross"]
    return {
        "factor_combo": combo,
        "sample": sample,
        "regime_group": regime_group,
        "months": len(frame),
        "strategy_annualized_return_net": strategy_cagr,
        "strategy_mdd_monthly": strategy_mdd,
        "kospi_annualized_return_next_close_proxy": kospi_cagr,
        "kospi_mdd_monthly_proxy": kospi_mdd,
        "equal_weight_annualized_return_gross": equal_weight_cagr,
        "equal_weight_mdd_monthly_gross": equal_weight_mdd,
        "excess_annualized_vs_kospi": strategy_cagr - kospi_cagr,
        "excess_annualized_vs_equal_weight": strategy_cagr - equal_weight_cagr,
        "mean_rank_ic": float(frame["rank_ic"].mean()),
        "rank_ic_nw_t": newey_west_t_stat(frame["rank_ic"]),
        "mean_monthly_excess_vs_kospi": float(excess_kospi.mean()),
        "excess_vs_kospi_nw_t": newey_west_t_stat(excess_kospi),
        "mean_monthly_excess_vs_equal_weight": float(excess_equal_weight.mean()),
        "excess_vs_equal_weight_nw_t": newey_west_t_stat(excess_equal_weight),
        "positive_month_ratio": float(frame["strategy_return_net"].gt(0).mean()),
        "median_selected_stocks": float(frame["selected_stocks"].median()),
        "passes_live_gate": False,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for combo, combo_frame in detail.groupby("factor_combo", sort=False):
        for sample, sample_frame in combo_frame.groupby("sample", sort=False):
            groups = {
                "all": sample_frame,
                "non_concentrated": sample_frame.loc[sample_frame["regime"].ne("concentrated")],
                "concentrated": sample_frame.loc[sample_frame["regime"].eq("concentrated")],
                "broad_bull": sample_frame.loc[sample_frame["regime"].eq("broad_bull")],
                "defensive": sample_frame.loc[sample_frame["regime"].eq("defensive")],
            }
            for group_name, group_frame in groups.items():
                if not group_frame.empty:
                    rows.append(summarize_group(group_frame, combo, sample, group_name))
    report = pd.DataFrame(rows)

    # This objective was stated after seeing the opened period, so it remains a
    # research-only label even though its inputs below are train-only.
    train_normal = report.loc[
        report["sample"].eq("train") & report["regime_group"].eq("non_concentrated")
    ].set_index("factor_combo")
    train_all = report.loc[
        report["sample"].eq("train") & report["regime_group"].eq("all")
    ].set_index("factor_combo")
    report["passes_normal_regime_research_gate_not_independent"] = False
    familywise_critical_z = NormalDist().inv_cdf(
        1 - FAMILYWISE_ALPHA / (2 * len(COMBINATIONS))
    )
    report["current_family_bonferroni_critical_z"] = familywise_critical_z
    for combo in train_normal.index.intersection(train_all.index):
        normal = train_normal.loc[combo]
        all_period = train_all.loc[combo]
        passed = bool(
            normal["months"] >= 24
            and normal["excess_annualized_vs_kospi"] >= 0.05
            and normal["excess_annualized_vs_equal_weight"] >= 0.05
            and normal["strategy_mdd_monthly"] >= -0.30
            and all_period["strategy_annualized_return_net"] > 0
            and normal["excess_vs_kospi_nw_t"] >= familywise_critical_z
            and normal["excess_vs_equal_weight_nw_t"] >= familywise_critical_z
        )
        report.loc[
            report["factor_combo"].eq(combo),
            "passes_normal_regime_research_gate_not_independent",
        ] = passed
    return report


def main() -> None:
    directions = require_train_selected_factors()
    print(f"SeedForge 030 수급 조합 시장상태 분해 (build {BUILD})")
    print("집중장세 규칙: 028 고정값 재사용, 종목명·날짜 수동 제외 없음")
    close, open_, volume, entry_eligible, kospi = core.load_data()
    flows = load_flow(find_flow_file(), close.columns)
    signal_dates = pd.DatetimeIndex(flows["외국인"].index)
    target = forward_target(close, open_, entry_eligible, signal_dates, HORIZON)
    flow_factor = build_factors(flows, close, volume)[FLOW_FACTOR]
    technical_builders = expanded_factor_builders(close, open_, volume)
    technical_names = {name for combo in COMBINATIONS for name in combo if name != FLOW_FACTOR}
    matrices: dict[str, pd.DataFrame] = {
        FLOW_FACTOR: oriented_rank(flow_factor, directions[FLOW_FACTOR])
    }
    for name in sorted(technical_names):
        daily = technical_builders[name][1]()
        matrices[name] = oriented_rank(asof_daily_matrix(daily, signal_dates), directions[name])

    regimes = fixed_regimes(close, entry_eligible, kospi, signal_dates)
    kospi_target = kospi_forward_proxy(kospi, signal_dates, HORIZON)
    rows: list[dict[str, object]] = []
    for number, combo in enumerate(COMBINATIONS, 1):
        name = " + ".join(combo)
        components = tuple(matrices[factor] for factor in combo)
        combo_rows = monthly_combo_rows(name, components, target, kospi_target, regimes)
        rows.extend(combo_rows)
        print(f"[{number:>2}/{len(COMBINATIONS)}] {name}: {len(combo_rows)}개월")
    detail = pd.DataFrame(rows).sort_values(["factor_combo", "date"])
    if detail.empty:
        raise RuntimeError("평가 가능한 030 월별 결과가 없습니다.")
    report = summarize(detail)
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_OUTPUT, index=False, encoding="utf-8-sig")
    report.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    display = report.loc[report["regime_group"].isin(["all", "non_concentrated", "concentrated"])]
    display = display.sort_values(
        ["sample", "regime_group", "excess_annualized_vs_kospi"],
        ascending=[False, False, False],
    )
    columns = [
        "factor_combo", "sample", "regime_group", "months", "strategy_annualized_return_net",
        "strategy_mdd_monthly", "kospi_annualized_return_next_close_proxy", "equal_weight_annualized_return_gross",
        "excess_annualized_vs_kospi", "excess_annualized_vs_equal_weight", "mean_rank_ic",
        "rank_ic_nw_t", "excess_vs_kospi_nw_t", "excess_vs_equal_weight_nw_t",
        "current_family_bonferroni_critical_z",
        "passes_normal_regime_research_gate_not_independent", "passes_live_gate",
    ]
    print("\nSeedForge 030 고정 시장상태별 결과")
    print(display[columns].to_string(index=False))
    print(f"\n월별 저장: {DETAIL_OUTPUT}")
    print(f"요약 저장: {SUMMARY_OUTPUT}")
    print("주의: KOSPI 시가가 없어 다음 거래일 종가 기준 proxy를 사용합니다.")
    print("주의: 정상장세 목표는 opened 확인 후 선언되었으므로 연구 진단일 뿐 독립 PASS가 아닙니다.")
    print("주의: Bonferroni 보정은 이번 9개 조합에만 적용하며 과거 전체 연구 횟수 보정은 아닙니다.")
    print("어떤 결과도 실전 승인으로 바꾸지 않습니다.")


if __name__ == "__main__":
    main()
