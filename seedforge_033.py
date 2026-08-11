"""SeedForge 033: exhaustive buy-rule search with the RSI sell module frozen.

This stage deliberately stops searching exit rules.  Every full portfolio run
uses the same point-in-time RSI policy: neutral bearish divergence exits the
position, overbought bearish divergence sells half, and a confirmed signal is
executed at the next tradable open.  The search varies only the buy equation
and portfolio construction.

The combinatorial universe can be very large.  Therefore the program is
restartable and uses two explicit stages:

1. Every eligible factor combination is evaluated by a cheap train-only proxy.
2. A configurable number of train-only finalists (or *all* combinations with
   ``--full-finalists 0``) receives the expensive RSI portfolio grid.

No opened-period value participates in factor direction, combination ranking,
job selection, or checkpoint resumption.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import seedforge_021 as core
import seedforge_031 as search
from seedforge_029 import find_flow_file, load_flow


BUILD = "033.0-20260811"
RESULTS = Path("data/results")
SCREEN_031 = RESULTS / "factor_screen_seedforge_031.csv"
PROXY_OUTPUT = RESULTS / "proxy_all_combos_seedforge_033.csv"
SUMMARY_OUTPUT = RESULTS / "summary_seedforge_033.csv"
MANIFEST_OUTPUT = RESULTS / "manifest_seedforge_033.json"
FINALISTS_OUTPUT = RESULTS / "finalists_seedforge_033.csv"

DISCOVERY_END = pd.Timestamp("2017-12-31")
VALIDATION_START = pd.Timestamp("2018-01-01")
VALIDATION_END = pd.Timestamp("2021-12-31")
OPENED_START = pd.Timestamp("2022-01-01")

DEFAULT_MAX_FACTORS = 4
# Zero is intentional: by default every train-eligible combination proceeds to
# the fixed-RSI portfolio grid.  Use --full-finalists N for a faster staged run.
DEFAULT_FULL_FINALISTS = 0
PORTFOLIO_SIZES = (20, 30, 50, 80)
REBALANCE_MONTHS = (1, 2, 3)
EXIT_RANK_MULTIPLIERS = (1.0, 1.5, 2.0)
WEIGHT_POLICIES = ("equal", "discovery_ic", "validation_ic")
CHECKPOINT_EVERY = 25


@dataclass(frozen=True)
class PortfolioConfig:
    size: int
    rebalance_months: int
    exit_rank_multiplier: float
    weight_policy: str


@dataclass
class PortfolioResult:
    equity: pd.Series
    exposure: pd.Series
    transaction_count: int
    turnover: float
    transaction_cost: float
    rsi_neutral_exits: int
    rsi_overbought_halves: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSI 매도를 고정한 SeedForge 033 대규모 매수식 탐색"
    )
    parser.add_argument(
        "--max-factors", type=int, default=DEFAULT_MAX_FACTORS, choices=range(1, 7),
        help="조합당 최대 팩터 수 (기본 4)",
    )
    parser.add_argument(
        "--full-finalists", type=int, default=DEFAULT_FULL_FINALISTS,
        help="실제 RSI 포트폴리오를 돌릴 proxy 상위 수. 0이면 모든 조합",
    )
    parser.add_argument(
        "--include-unstable-singles", action="store_true",
        help="031 train_stable 미통과 단독 팩터까지 포함 (권장하지 않음)",
    )
    parser.add_argument(
        "--same-group", action="store_true",
        help="같은 계열 팩터끼리의 조합도 포함하여 진짜 전 조합 실행",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="이번 실행의 포트폴리오 작업 수 제한. 0이면 제한 없음",
    )
    parser.add_argument("--restart", action="store_true", help="기존 033 결과를 지우고 재시작")
    return parser.parse_args()


def combo_key(combo: tuple[str, ...]) -> str:
    return " + ".join(combo)


def job_id(combo: tuple[str, ...], config: PortfolioConfig) -> str:
    text = "|".join((
        combo_key(combo), str(config.size), str(config.rebalance_months),
        str(config.exit_rank_multiplier), config.weight_policy,
    ))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_single_universe(include_unstable: bool) -> pd.DataFrame:
    if not SCREEN_031.exists():
        raise FileNotFoundError(
            "031 단독 팩터 결과가 없습니다. 먼저 python -u seedforge_031.py 를 실행하세요."
        )
    singles = pd.read_csv(SCREEN_031)
    required = {
        "factor", "group", "direction", "train_stable", "discovery_mean_ic",
        "validation_mean_ic", "search_score",
    }
    missing = required.difference(singles.columns)
    if missing:
        raise ValueError(f"031 단독 결과 필수 열이 없습니다: {sorted(missing)}")
    if not include_unstable:
        singles = singles.loc[singles["train_stable"].astype(bool)]
    singles = singles.sort_values(
        ["train_stable", "search_score", "factor"], ascending=[False, False, True]
    ).drop_duplicates("factor")
    if singles.empty:
        raise RuntimeError("탐색에 사용할 train-only 단독 팩터가 없습니다.")
    return singles.reset_index(drop=True)


def enumerate_combos(
    singles: pd.DataFrame,
    max_factors: int,
    same_group: bool,
) -> Iterable[tuple[str, ...]]:
    names = singles["factor"].astype(str).tolist()
    groups = singles.set_index("factor")["group"].astype(str).to_dict()
    for size in range(1, min(max_factors, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            if not same_group and len({groups[name] for name in combo}) != size:
                continue
            yield tuple(sorted(combo))


def combination_count(singles: pd.DataFrame, max_factors: int, same_group: bool) -> int:
    return sum(1 for _ in enumerate_combos(singles, max_factors, same_group))


def build_oriented_library(
    singles: pd.DataFrame,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    flows = load_flow(find_flow_file(), close.columns)
    raw, groups = search.build_factor_library(close, open_, volume, dates, flows)
    directions = singles.set_index("factor")["direction"].astype(int).to_dict()
    missing = set(directions).difference(raw)
    if missing:
        raise ValueError(f"팩터 라이브러리에서 누락된 031 팩터: {sorted(missing)}")
    oriented = {
        name: (raw[name] if direction > 0 else 1 - raw[name]).astype(np.float32, copy=False)
        for name, direction in directions.items()
    }
    return oriented, {name: groups[name] for name in oriented}


def proxy_row(
    combo: tuple[str, ...],
    oriented: dict[str, np.ndarray],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, object]:
    score = np.nanmean(np.stack([oriented[name] for name in combo]), axis=0)
    result = search.evaluate_score(score, target, dates, include_opened=False)
    return {
        "factor_combo": combo_key(combo),
        "factor_count": len(combo),
        **result,
        "opened_used_for_selection": False,
        "passes_live_gate": False,
    }


def run_proxy_universe(
    combos: list[tuple[str, ...]],
    oriented: dict[str, np.ndarray],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
    restart: bool,
) -> pd.DataFrame:
    if PROXY_OUTPUT.exists() and not restart:
        cached = pd.read_csv(PROXY_OUTPUT)
        expected = {combo_key(combo) for combo in combos}
        if expected.issubset(set(cached["factor_combo"].astype(str))):
            print(f"proxy 전체 결과 재사용: {PROXY_OUTPUT}")
            return cached.loc[cached["factor_combo"].isin(expected)].copy()

    rows: list[dict[str, object]] = []
    total = len(combos)
    print(f"\n1단계: train-only proxy 전 조합 {total:,}개")
    for number, combo in enumerate(combos, 1):
        rows.append(proxy_row(combo, oriented, target, dates))
        if number % 100 == 0 or number == total:
            print(f"  proxy {number:,}/{total:,}")
    report = pd.DataFrame(rows).sort_values(
        ["train_stable", "search_score", "robust_ic"], ascending=False
    )
    report.to_csv(PROXY_OUTPUT, index=False, encoding="utf-8-sig")
    return report


def factor_weights(
    combo: tuple[str, ...], singles: pd.DataFrame, policy: str
) -> np.ndarray:
    indexed = singles.set_index("factor")
    if policy == "equal":
        values = np.ones(len(combo), dtype=float)
    elif policy == "discovery_ic":
        values = indexed.loc[list(combo), "discovery_mean_ic"].to_numpy(dtype=float)
    elif policy == "validation_ic":
        values = indexed.loc[list(combo), "validation_mean_ic"].to_numpy(dtype=float)
    else:
        raise ValueError(f"알 수 없는 가중 방식: {policy}")
    values = np.clip(values, 0, None)
    return values / values.sum() if values.sum() > 0 else np.ones(len(combo)) / len(combo)


def monthly_rankings(
    combo: tuple[str, ...],
    policy: str,
    singles: pd.DataFrame,
    oriented: dict[str, np.ndarray],
    signal_dates: pd.DatetimeIndex,
    market: core.SimulationInputs,
    max_exit_rank: int,
) -> dict[int, tuple[int, ...]]:
    weights = factor_weights(combo, singles, policy)
    score = np.zeros_like(oriented[combo[0]], dtype=np.float32)
    valid_weight = np.zeros_like(score, dtype=np.float32)
    for name, weight in zip(combo, weights):
        component = oriented[name]
        valid = np.isfinite(component)
        score[valid] += component[valid] * weight
        valid_weight[valid] += weight
    score = np.divide(score, valid_weight, out=np.full_like(score, np.nan), where=valid_weight > 0)

    rankings: dict[int, tuple[int, ...]] = {}
    for row, signal_date in enumerate(signal_dates):
        source_day = int(market.dates.searchsorted(signal_date, side="right") - 1)
        execution_day = source_day + 1
        if source_day < 0 or execution_day >= len(market.dates):
            continue
        values = score[row]
        valid = (
            np.isfinite(values)
            & market.entry_eligible[execution_day]
            & np.isfinite(market.open[execution_day])
            & (market.open[execution_day] > 0)
        )
        candidates = np.flatnonzero(valid)
        if not candidates.size:
            continue
        count = min(max_exit_rank, len(candidates))
        selected = candidates[np.argpartition(values[candidates], -count)[-count:]]
        selected = selected[np.argsort(values[selected])[::-1]]
        rankings[execution_day] = tuple(int(stock) for stock in selected)
    return rankings


def simulate_fixed_rsi(
    market: core.SimulationInputs,
    rankings: dict[int, tuple[int, ...]],
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    config: PortfolioConfig,
) -> PortfolioResult:
    cash = float(core.INITIAL_CAPITAL)
    shares: dict[int, float] = {}
    equity: list[float] = []
    exposure: list[float] = []
    transaction_count = 0
    traded_gross = transaction_cost = 0.0
    neutral_exits = overbought_halves = 0

    def sell(day: int, stock: int, fraction: float) -> bool:
        nonlocal cash, transaction_count, traded_gross, transaction_cost
        held = shares.get(stock, 0.0)
        price = market.open[day, stock]
        if held <= 0 or not (np.isfinite(price) and price > 0):
            return False
        quantity = held * min(max(fraction, 0.0), 1.0)
        gross = quantity * price
        fee = gross * core.COST_PER_SIDE
        cash += gross - fee
        remaining = held - quantity
        if remaining <= held * 1e-10:
            shares.pop(stock, None)
        else:
            shares[stock] = remaining
        transaction_count += 1
        traded_gross += gross
        transaction_cost += fee
        return True

    def buy(day: int, stock: int, desired: float) -> None:
        nonlocal cash, transaction_count, traded_gross, transaction_cost
        price = market.open[day, stock]
        if desired <= 0 or not market.entry_eligible[day, stock]:
            return
        if not (np.isfinite(price) and price > 0):
            return
        gross = min(desired, cash / (1 + core.COST_PER_SIDE))
        if gross <= 0:
            return
        fee = gross * core.COST_PER_SIDE
        shares[stock] = shares.get(stock, 0.0) + gross / price
        cash -= gross + fee
        transaction_count += 1
        traded_gross += gross
        transaction_cost += fee

    rebalance_number = 0
    for day in range(len(market.dates)):
        blocked: set[int] = set()
        if day > 0:
            neutral_today = neutral.get(day - 1, set())
            overbought_today = overbought.get(day - 1, set()) - neutral_today
            for stock in tuple(shares):
                if stock in neutral_today and sell(day, stock, 1.0):
                    neutral_exits += 1
                    blocked.add(stock)
                elif stock in overbought_today and sell(day, stock, 0.5):
                    overbought_halves += 1
                    blocked.add(stock)

        ranking = rankings.get(day)
        if ranking is not None:
            execute_rebalance = rebalance_number % config.rebalance_months == 0
            rebalance_number += 1
            if execute_rebalance:
                exit_rank = max(config.size, int(np.ceil(config.size * config.exit_rank_multiplier)))
                allowed = set(ranking[:exit_rank]) - blocked
                retained = [stock for stock in shares if stock in allowed]
                target = retained[:config.size]
                for stock in ranking:
                    if len(target) >= config.size:
                        break
                    if stock not in target and stock not in blocked:
                        target.append(stock)
                target_set = set(target)
                for stock in tuple(shares):
                    if stock not in target_set:
                        sell(day, stock, 1.0)

                opening_equity = cash + sum(
                    quantity * market.open[day, stock]
                    for stock, quantity in shares.items()
                    if np.isfinite(market.open[day, stock])
                )
                target_value = opening_equity / max(len(target), 1)
                for stock in target:
                    value = shares.get(stock, 0.0) * market.open[day, stock]
                    if np.isfinite(value) and value > target_value:
                        sell(day, stock, (value - target_value) / value)
                for stock in target:
                    value = shares.get(stock, 0.0) * market.open[day, stock]
                    if np.isfinite(value) and value < target_value:
                        buy(day, stock, target_value - value)

        marked = sum(
            quantity * market.marked_close[day, stock]
            for stock, quantity in shares.items()
            if np.isfinite(market.marked_close[day, stock])
        )
        total = cash + marked
        equity.append(total)
        exposure.append(marked / total if total > 0 else 0.0)

    return PortfolioResult(
        equity=pd.Series(equity, index=market.dates),
        exposure=pd.Series(exposure, index=market.dates),
        transaction_count=transaction_count,
        turnover=traded_gross / (2 * core.INITIAL_CAPITAL),
        transaction_cost=transaction_cost,
        rsi_neutral_exits=neutral_exits,
        rsi_overbought_halves=overbought_halves,
    )


def period_metrics(equity: pd.Series, start: str | None, end: str | None) -> dict[str, float]:
    values = equity.loc[start:end].dropna() if start or end else equity.dropna()
    if len(values) < 2 or values.iloc[0] <= 0:
        return {"cagr": np.nan, "mdd": np.nan, "calmar": np.nan, "two_year_return": np.nan}
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
    mdd = float((values / values.cummax() - 1).min())
    return {
        "cagr": cagr,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "two_year_return": (1 + cagr) ** 2 - 1,
    }


def summarize_job(
    identifier: str,
    combo: tuple[str, ...],
    config: PortfolioConfig,
    proxy_rank: int,
    result: PortfolioResult,
) -> dict[str, object]:
    row: dict[str, object] = {
        "job_id": identifier, "build": BUILD, "proxy_train_rank": proxy_rank,
        "factor_combo": combo_key(combo), "factor_count": len(combo),
        "portfolio_size": config.size, "rebalance_months": config.rebalance_months,
        "exit_rank_multiplier": config.exit_rank_multiplier,
        "weight_policy": config.weight_policy,
        "exit_policy": "fixed_neutral_plus_overbought",
        "transactions": result.transaction_count, "turnover": result.turnover,
        "transaction_cost": result.transaction_cost,
        "rsi_neutral_full_exits": result.rsi_neutral_exits,
        "rsi_overbought_half_exits": result.rsi_overbought_halves,
        "average_exposure": float(result.exposure.mean()),
        "opened_used_for_selection": False, "passes_live_gate": False,
    }
    for suffix, start, end in (
        ("discovery", None, "2017-12-31"),
        ("validation", "2018-01-01", "2021-12-31"),
        ("train", None, "2021-12-31"),
        ("opened", "2022-01-01", None),
        ("full", None, None),
    ):
        metrics = period_metrics(result.equity, start, end)
        row.update({f"{name}_{suffix}": value for name, value in metrics.items()})
    row["robust_train_cagr"] = min(row["cagr_discovery"], row["cagr_validation"])
    row["robust_train_calmar"] = min(row["calmar_discovery"], row["calmar_validation"])
    row["train_selection_score"] = (
        row["robust_train_cagr"]
        + 0.25 * row["cagr_validation"]
        + 0.05 * row["robust_train_calmar"]
        - 0.02 * row["turnover"] / 10
    )
    return row


def configurations() -> list[PortfolioConfig]:
    return [
        PortfolioConfig(size, months, multiplier, weight)
        for size, months, multiplier, weight in itertools.product(
            PORTFOLIO_SIZES, REBALANCE_MONTHS, EXIT_RANK_MULTIPLIERS, WEIGHT_POLICIES
        )
    ]


def completed_jobs(restart: bool) -> tuple[set[str], list[dict[str, object]]]:
    if restart or not SUMMARY_OUTPUT.exists():
        return set(), []
    existing = pd.read_csv(SUMMARY_OUTPUT)
    return set(existing["job_id"].astype(str)), existing.to_dict("records")


def save_summary(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).drop_duplicates("job_id", keep="last").to_csv(
        SUMMARY_OUTPUT, index=False, encoding="utf-8-sig"
    )


def write_manifest(
    args: argparse.Namespace,
    singles: pd.DataFrame,
    combo_total: int,
    finalist_total: int,
    job_total: int,
) -> None:
    payload = {
        "build": BUILD,
        "single_factors": len(singles),
        "max_factors": args.max_factors,
        "same_group_combinations": args.same_group,
        "all_proxy_combinations": combo_total,
        "full_portfolio_finalists": finalist_total,
        "portfolio_configurations_per_combo": len(configurations()),
        "full_portfolio_jobs": job_total,
        "portfolio_sizes": PORTFOLIO_SIZES,
        "rebalance_months": REBALANCE_MONTHS,
        "exit_rank_multipliers": EXIT_RANK_MULTIPLIERS,
        "weight_policies": WEIGHT_POLICIES,
        "exit_policy": "fixed_neutral_plus_overbought",
        "opened_used_for_selection": False,
        "passes_live_gate": False,
    }
    MANIFEST_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_resume_manifest(
    args: argparse.Namespace,
    singles: pd.DataFrame,
    combo_total: int,
) -> None:
    """Prevent checkpoints from different search universes being mixed."""
    if args.restart or not SUMMARY_OUTPUT.exists():
        return
    if not MANIFEST_OUTPUT.exists():
        raise RuntimeError("033 요약은 있지만 manifest가 없습니다. --restart로 다시 시작하세요.")
    previous = json.loads(MANIFEST_OUTPUT.read_text(encoding="utf-8"))
    expected = {
        "build": BUILD,
        "single_factors": len(singles),
        "max_factors": args.max_factors,
        "same_group_combinations": args.same_group,
        "all_proxy_combinations": combo_total,
        "portfolio_sizes": list(PORTFOLIO_SIZES),
        "rebalance_months": list(REBALANCE_MONTHS),
        "exit_rank_multipliers": list(EXIT_RANK_MULTIPLIERS),
        "weight_policies": list(WEIGHT_POLICIES),
    }
    mismatches = {
        key: (previous.get(key), value)
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "기존 033 checkpoint와 현재 탐색 범위가 다릅니다. "
            f"--restart가 필요합니다: {mismatches}"
        )


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for path in (PROXY_OUTPUT, SUMMARY_OUTPUT, MANIFEST_OUTPUT, FINALISTS_OUTPUT):
            path.unlink(missing_ok=True)

    print(f"SeedForge 033 RSI 고정 대규모 매수식 탐색 (build {BUILD})")
    print("매도는 고정: 중립 다이버전스 전량 / 과매수 다이버전스 50%")
    singles = load_single_universe(args.include_unstable_singles)
    combo_total = combination_count(singles, args.max_factors, args.same_group)
    validate_resume_manifest(args, singles, combo_total)
    print(f"단독 팩터 {len(singles):,}개, proxy 전 조합 {combo_total:,}개")

    market, _features, overbought, neutral, _kospi = core.load_or_prepare(False)
    close = pd.DataFrame(market.close, index=market.dates, columns=market.tickers)
    open_ = pd.DataFrame(market.open, index=market.dates, columns=market.tickers)
    volume = pd.DataFrame(market.volume, index=market.dates, columns=market.tickers)
    flows = load_flow(find_flow_file(), close.columns)
    signal_dates = pd.DatetimeIndex(flows["외국인"].index)
    oriented, _groups = build_oriented_library(singles, close, open_, volume, signal_dates)
    target = search.forward_target(
        close, open_, pd.DataFrame(market.entry_eligible, index=market.dates, columns=market.tickers),
        signal_dates, search.HORIZON_DAYS,
    ).to_numpy(dtype=np.float32)
    combos = list(enumerate_combos(singles, args.max_factors, args.same_group))
    proxy = run_proxy_universe(combos, oriented, target, signal_dates, args.restart)

    ranked = proxy.sort_values(
        ["train_stable", "search_score", "robust_ic"], ascending=False
    ).reset_index(drop=True)
    if args.full_finalists < 0:
        raise ValueError("--full-finalists는 0 이상이어야 합니다.")
    finalists = ranked if args.full_finalists == 0 else ranked.head(args.full_finalists)
    finalists.to_csv(FINALISTS_OUTPUT, index=False, encoding="utf-8-sig")
    configs = configurations()
    total_jobs = len(finalists) * len(configs)
    write_manifest(args, singles, combo_total, len(finalists), total_jobs)
    print(
        f"\n2단계: RSI 고정 실제 포트폴리오 {len(finalists):,}조합 × "
        f"{len(configs)}설정 = {total_jobs:,}작업"
    )
    if args.full_finalists == 0:
        print("경고: 모든 조합 full 모드는 매우 오래 걸립니다. 중단 후 같은 명령으로 이어집니다.")

    done, rows = completed_jobs(args.restart)
    executed_this_run = 0
    for proxy_rank, finalist in enumerate(finalists.itertuples(index=False), 1):
        combo = tuple(str(finalist.factor_combo).split(" + "))
        max_exit_rank = int(max(PORTFOLIO_SIZES) * max(EXIT_RANK_MULTIPLIERS))
        ranking_cache = {
            policy: monthly_rankings(
                combo, policy, singles, oriented, signal_dates, market, max_exit_rank
            )
            for policy in WEIGHT_POLICIES
        }
        for config in configs:
            identifier = job_id(combo, config)
            if identifier in done:
                continue
            result = simulate_fixed_rsi(
                market, ranking_cache[config.weight_policy], overbought, neutral, config
            )
            rows.append(summarize_job(identifier, combo, config, proxy_rank, result))
            done.add(identifier)
            executed_this_run += 1
            completed = len(done)
            if executed_this_run % CHECKPOINT_EVERY == 0:
                save_summary(rows)
                print(f"  checkpoint 완료 {completed:,}/{total_jobs:,}")
            if args.max_jobs and executed_this_run >= args.max_jobs:
                save_summary(rows)
                print("이번 --max-jobs 한도에 도달했습니다. 같은 명령으로 재실행하면 이어집니다.")
                return

    save_summary(rows)
    result_frame = pd.DataFrame(rows).sort_values(
        ["train_selection_score", "robust_train_cagr", "robust_train_calmar"],
        ascending=False,
    )
    result_frame.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    display = [
        "factor_combo", "portfolio_size", "rebalance_months", "exit_rank_multiplier",
        "weight_policy", "cagr_train", "mdd_train", "calmar_train",
        "robust_train_cagr", "cagr_opened", "mdd_opened", "passes_live_gate",
    ]
    print("\nRSI 고정 train-only 상위 설정")
    print(result_frame[display].head(30).to_string(index=False))
    print(f"\n전 조합 proxy: {PROXY_OUTPUT}")
    print(f"실제 포트폴리오: {SUMMARY_OUTPUT}")
    print(f"실행 원장: {MANIFEST_OUTPUT}")
    print("주의: opened 열은 최종 출력 진단일 뿐 순위·선택·재개 판단에 사용하지 않습니다.")
    print("주의: 어떤 결과도 신규 전진검증 전에는 실전 PASS가 아닙니다.")


if __name__ == "__main__":
    main()
