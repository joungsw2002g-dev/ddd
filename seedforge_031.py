"""SeedForge 031: broad, staged buy-condition search without opened-data selection.

The search evaluates many technical, flow, and point-in-time DART factors, but
controls the explosion with correlation pruning and a cross-family beam search.
Directions are learned on 2014-2017, combinations are selected on 2018-2021,
and 2022 onward is evaluated only after finalists have been frozen.
"""

from __future__ import annotations

import gc
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import seedforge_021 as core
from seedforge_025_screen import expanded_factor_builders
from seedforge_026 import dart_factor_columns, load_dart_events, point_in_time_matrix
from seedforge_029 import build_factors, find_flow_file, forward_target, load_flow


BUILD = "031.0-20260811"
RESULTS = Path("data/results")
SINGLE_OUTPUT = RESULTS / "factor_screen_seedforge_031.csv"
COMBO_OUTPUT = RESULTS / "factor_combos_seedforge_031.csv"
FINALIST_OUTPUT = RESULTS / "opened_diagnostic_seedforge_031.csv"
MANIFEST_OUTPUT = RESULTS / "search_manifest_seedforge_031.csv"

DISCOVERY_END = pd.Timestamp("2017-12-31")
VALIDATION_START = pd.Timestamp("2018-01-01")
VALIDATION_END = pd.Timestamp("2021-12-31")
OPENED_START = pd.Timestamp("2022-01-01")
HORIZON_DAYS = 20
MIN_STOCKS = 80
MIN_DISCOVERY_MONTHS = 24
MIN_VALIDATION_MONTHS = 36
MAX_PER_GROUP = 3
MAX_PAIR_CORRELATION = 0.85
MAX_FACTORS = 4
BEAM_WIDTH = 30
FINALISTS = 20
FAMILYWISE_ALPHA = 0.05
NW_LAGS = 3


def asof_daily_matrix(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if not daily.index.is_monotonic_increasing:
        daily = daily.sort_index()
    if daily.index.has_duplicates:
        raise ValueError("일별 팩터 날짜가 중복되었습니다.")
    union = daily.index.union(dates).sort_values()
    return daily.reindex(union).ffill().reindex(dates)


def rank_array(frame: pd.DataFrame) -> np.ndarray:
    return frame.rank(axis=1, pct=True).to_numpy(dtype=np.float32)


def newey_west_t(values: list[float], lags: int = NW_LAGS) -> float:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
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


def period_metrics(
    score: np.ndarray,
    target: np.ndarray,
    dates: pd.DatetimeIndex,
    mask: np.ndarray,
) -> dict[str, float | int]:
    ics: list[float] = []
    spreads: list[float] = []
    selected_excess: list[float] = []
    selected_returns: list[float] = []
    observations: list[int] = []
    for row in np.flatnonzero(mask):
        x, y = score[row], target[row]
        valid = np.isfinite(x) & np.isfinite(y)
        count = int(valid.sum())
        if count < MIN_STOCKS:
            continue
        xv, yv = x[valid], y[valid]
        if np.nanmax(xv) <= np.nanmin(xv):
            continue
        y_rank = pd.Series(yv).rank(pct=True).to_numpy(dtype=float)
        ic = float(np.corrcoef(xv, y_rank)[0, 1])
        threshold_high = np.nanquantile(xv, 0.8)
        threshold_low = np.nanquantile(xv, 0.2)
        top = yv[xv >= threshold_high]
        bottom = yv[xv <= threshold_low]
        selected = float(np.nanmean(top))
        equal_weight = float(np.nanmean(yv))
        ics.append(ic)
        spreads.append(float(np.nanmean(top) - np.nanmean(bottom)))
        selected_returns.append(selected)
        selected_excess.append(selected - equal_weight)
        observations.append(count)
    return {
        "months": len(ics),
        "mean_ic": float(np.nanmean(ics)) if ics else np.nan,
        "ic_nw_t": newey_west_t(ics),
        "mean_spread": float(np.nanmean(spreads)) if spreads else np.nan,
        "mean_selected_return_gross": (
            float(np.nanmean(selected_returns)) if selected_returns else np.nan
        ),
        "mean_selected_excess_gross": (
            float(np.nanmean(selected_excess)) if selected_excess else np.nan
        ),
        "selected_excess_nw_t": newey_west_t(selected_excess),
        "median_observations": float(np.nanmedian(observations)) if observations else np.nan,
    }


def evaluate_score(
    score: np.ndarray,
    target: np.ndarray,
    dates: pd.DatetimeIndex,
    include_opened: bool,
) -> dict[str, object]:
    discovery_mask = np.asarray(dates <= DISCOVERY_END)
    validation_mask = np.asarray(
        (dates >= VALIDATION_START) & (dates <= VALIDATION_END)
    )
    discovery = period_metrics(score, target, dates, discovery_mask)
    validation = period_metrics(score, target, dates, validation_mask)
    row: dict[str, object] = {}
    for prefix, metrics in (("discovery", discovery), ("validation", validation)):
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    robust_ic = min(float(discovery["mean_ic"]), float(validation["mean_ic"]))
    # IC dominates; gross excess and spread break ties without embedding a cost guess.
    row["robust_ic"] = robust_ic
    row["search_score"] = (
        robust_ic
        + 0.25 * float(validation["mean_ic"])
        + 0.25 * float(validation["mean_selected_excess_gross"])
        + 0.10 * float(validation["mean_spread"])
    )
    row["train_stable"] = bool(
        discovery["months"] >= MIN_DISCOVERY_MONTHS
        and validation["months"] >= MIN_VALIDATION_MONTHS
        and discovery["mean_ic"] > 0
        and validation["mean_ic"] > 0
        and discovery["mean_spread"] > 0
        and validation["mean_spread"] > 0
        and validation["mean_selected_excess_gross"] > 0
    )
    if include_opened:
        opened = period_metrics(score, target, dates, np.asarray(dates >= OPENED_START))
        row.update({f"opened_{key}": value for key, value in opened.items()})
    return row


def mean_abs_rank_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    correlations: list[float] = []
    for row in np.flatnonzero(mask):
        valid = np.isfinite(left[row]) & np.isfinite(right[row])
        if int(valid.sum()) < MIN_STOCKS:
            continue
        value = float(np.corrcoef(left[row, valid], right[row, valid])[0, 1])
        if np.isfinite(value):
            correlations.append(abs(value))
    return float(np.mean(correlations)) if correlations else np.nan


def build_factor_library(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
    flows: dict[str, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}

    technical = expanded_factor_builders(close, open_, volume)
    print(f"기술 팩터 {len(technical)}개 준비 중...")
    for number, (name, (group, builder)) in enumerate(technical.items(), 1):
        monthly = asof_daily_matrix(builder(), dates)
        arrays[name] = rank_array(monthly)
        groups[name] = group
        del monthly
        if number % 10 == 0 or number == len(technical):
            print(f"  기술 {number}/{len(technical)}")
        gc.collect()

    flow_factors = build_factors(flows, close, volume)
    print(f"수급 팩터 {len(flow_factors)}개 준비 중...")
    for name, frame in flow_factors.items():
        arrays[name] = rank_array(frame.reindex(dates))
        groups[name] = "flow"

    try:
        events = load_dart_events()
    except FileNotFoundError:
        print("DART 파일 없음: DART 계열은 검색에서 제외합니다.")
    else:
        dart_columns = dart_factor_columns(events)
        print(f"DART 팩터 {len(dart_columns)}개 준비 중...")
        for name, values in dart_columns.items():
            matrix = point_in_time_matrix(events, values, dates, close.columns)
            arrays[name] = rank_array(matrix)
            groups[name] = "dart"
            del matrix
            gc.collect()
    return arrays, groups


def orient_and_screen_singles(
    arrays: dict[str, np.ndarray],
    groups: dict[str, str],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    discovery_mask = np.asarray(dates <= DISCOVERY_END)
    oriented: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for number, (name, raw) in enumerate(arrays.items(), 1):
        raw_metrics = period_metrics(raw, target, dates, discovery_mask)
        direction = 1 if float(raw_metrics["mean_ic"]) >= 0 else -1
        score = raw if direction > 0 else 1 - raw
        oriented[name] = score.astype(np.float32, copy=False)
        result = evaluate_score(score, target, dates, include_opened=False)
        result.update({"factor": name, "group": groups[name], "direction": direction})
        rows.append(result)
        if number % 10 == 0 or number == len(arrays):
            print(f"단독 스크린 {number}/{len(arrays)}")
    report = pd.DataFrame(rows).sort_values(
        ["train_stable", "search_score", "robust_ic"], ascending=False
    )
    return oriented, report


def prune_candidates(
    singles: pd.DataFrame,
    oriented: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> tuple[list[str], list[dict[str, object]]]:
    pool = (
        singles.loc[singles["train_stable"]]
        .sort_values(["group", "search_score"], ascending=[True, False])
        .groupby("group", group_keys=False)
        .head(MAX_PER_GROUP)
        .sort_values("search_score", ascending=False)
    )
    validation_mask = np.asarray(
        (dates >= VALIDATION_START) & (dates <= VALIDATION_END)
    )
    kept: list[str] = []
    audit: list[dict[str, object]] = []
    for row in pool.itertuples(index=False):
        blocker = ""
        correlation = np.nan
        for existing in kept:
            value = mean_abs_rank_correlation(
                oriented[row.factor], oriented[existing], validation_mask
            )
            if np.isfinite(value) and value >= MAX_PAIR_CORRELATION:
                blocker, correlation = existing, value
                break
        if not blocker:
            kept.append(row.factor)
        audit.append({
            "factor": row.factor,
            "group": row.group,
            "kept": not blocker,
            "blocked_by": blocker,
            "mean_abs_validation_rank_correlation": correlation,
        })
    return kept, audit


def beam_search(
    names: list[str],
    oriented: dict[str, np.ndarray],
    groups: dict[str, str],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int]:
    cache: dict[tuple[str, ...], dict[str, object]] = {}
    tested = 0

    def score_key(combo: tuple[str, ...]) -> float:
        value = float(cache[combo]["search_score"])
        return value if np.isfinite(value) else -np.inf

    def evaluate(combo: tuple[str, ...]) -> dict[str, object]:
        nonlocal tested
        if combo in cache:
            return cache[combo]
        score = np.nanmean(np.stack([oriented[name] for name in combo]), axis=0)
        result = evaluate_score(score, target, dates, include_opened=False)
        result.update({
            "factor_combo": " + ".join(combo),
            "factor_count": len(combo),
            "factor_groups": "+".join(groups[name] for name in combo),
            "selected_on_opened": False,
        })
        cache[combo] = result
        tested += 1
        return result

    beam = [(name,) for name in names]
    for combo in beam:
        evaluate(combo)
    beam = sorted(
        beam, key=score_key, reverse=True
    )[:BEAM_WIDTH]
    for depth in range(2, MAX_FACTORS + 1):
        candidates: set[tuple[str, ...]] = set()
        for combo in beam:
            used_groups = {groups[name] for name in combo}
            for name in names:
                if name in combo or groups[name] in used_groups:
                    continue
                candidates.add(tuple(sorted((*combo, name))))
        for combo in candidates:
            evaluate(combo)
        stable = [combo for combo in candidates if bool(cache[combo]["train_stable"])]
        source = stable if stable else list(candidates)
        beam = sorted(
            source, key=score_key, reverse=True
        )[:BEAM_WIDTH]
        print(
            f"깊이 {depth}: 신규 {len(candidates):,}, 안정 {len(stable):,}, "
            f"beam {len(beam):,}"
        )
        if not beam:
            break
    report = pd.DataFrame(cache.values()).sort_values(
        ["train_stable", "search_score", "robust_ic"], ascending=False
    )
    return report, tested


def add_opened_diagnostics(
    train_report: pd.DataFrame,
    oriented: dict[str, np.ndarray],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    finalists = train_report.head(FINALISTS).copy()
    rows: list[dict[str, object]] = []
    for row in finalists.itertuples(index=False):
        names = tuple(str(row.factor_combo).split(" + "))
        score = np.nanmean(np.stack([oriented[name] for name in names]), axis=0)
        result = evaluate_score(score, target, dates, include_opened=True)
        result.update({
            "factor_combo": row.factor_combo,
            "factor_count": row.factor_count,
            "factor_groups": row.factor_groups,
            "frozen_rank_from_train": len(rows) + 1,
            "selected_on_opened": False,
            "passes_live_gate": False,
        })
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"SeedForge 031 광범위 매수조건 탐색 (build {BUILD})")
    print("방향: 2014~2017 / 조합선택: 2018~2021 / opened: 최종후보 진단만")
    close, open_, volume, entry_eligible, _kospi = core.load_data()
    flows = load_flow(find_flow_file(), close.columns)
    dates = pd.DatetimeIndex(flows["외국인"].index)
    target_frame = forward_target(
        close, open_, entry_eligible, dates, HORIZON_DAYS
    )
    target = target_frame.to_numpy(dtype=np.float32)
    arrays, groups = build_factor_library(close, open_, volume, dates, flows)
    oriented, singles = orient_and_screen_singles(arrays, groups, target, dates)
    del arrays
    gc.collect()
    names, correlation_audit = prune_candidates(singles, oriented, dates)
    if len(names) < 2:
        raise RuntimeError("안정성·상관 제거 후 조합 후보가 2개 미만입니다.")
    print(f"단독 {len(singles)}개 → 조합 풀 {len(names)}개")
    combos, tested = beam_search(names, oriented, groups, target, dates)
    finalists = add_opened_diagnostics(combos, oriented, target, dates)

    critical_z = NormalDist().inv_cdf(1 - FAMILYWISE_ALPHA / (2 * max(tested, 1)))
    combos["tested_candidate_count"] = tested
    combos["search_family_bonferroni_critical_z"] = critical_z
    combos["passes_search_significance_gate"] = (
        combos["train_stable"]
        & combos["validation_ic_nw_t"].ge(critical_z)
        & combos["validation_selected_excess_nw_t"].ge(critical_z)
    )
    combos["passes_live_gate"] = False

    RESULTS.mkdir(parents=True, exist_ok=True)
    singles.to_csv(SINGLE_OUTPUT, index=False, encoding="utf-8-sig")
    combos.to_csv(COMBO_OUTPUT, index=False, encoding="utf-8-sig")
    finalists.to_csv(FINALIST_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(correlation_audit).to_csv(
        RESULTS / "factor_correlation_audit_seedforge_031.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "build": BUILD,
        "raw_factors": len(singles),
        "pruned_factors": len(names),
        "tested_candidates": tested,
        "beam_width": BEAM_WIDTH,
        "max_factors": MAX_FACTORS,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_critical_z": critical_z,
        "opened_used_for_selection": False,
    }]).to_csv(MANIFEST_OUTPUT, index=False, encoding="utf-8-sig")

    columns = [
        "factor_combo", "factor_groups", "factor_count", "discovery_mean_ic",
        "validation_mean_ic", "robust_ic", "validation_mean_spread",
        "validation_mean_selected_excess_gross", "validation_ic_nw_t",
        "validation_selected_excess_nw_t", "search_score", "train_stable",
        "passes_search_significance_gate", "passes_live_gate",
    ]
    print("\nTrain-only 상위 매수조건")
    print(combos[columns].head(30).to_string(index=False))
    print(f"\n단독 저장: {SINGLE_OUTPUT}")
    print(f"조합 저장: {COMBO_OUTPUT}")
    print(f"opened 진단 저장: {FINALIST_OUTPUT}")
    print(f"탐색 원장 저장: {MANIFEST_OUTPUT}")
    print(f"총 평가 후보: {tested:,}, Bonferroni 임계 z: {critical_z:.3f}")
    print("주의: opened 결과는 상위 후보를 고른 뒤에만 계산하며 실전 PASS는 항상 False입니다.")


if __name__ == "__main__":
    main()
