"""SeedForge 032: pair SeedForge 031 buy rules with/without RSI partial exits.

The control and overlay use the exact same monthly top-N targets and the same
rank-rebalance engine.  This avoids the invalid comparison where a no-exit
portfolio fills its slots forever while an RSI portfolio can recycle capital.
Only the overlay adds the previously selected ``neutral_plus_overbought`` rule:
neutral bearish divergence sells all and overbought bearish divergence sells
half.  Every close-confirmed signal executes at the next tradable open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
import seedforge_031 as search
from seedforge_029 import find_flow_file, load_flow


BUILD = "032.0-20260811"
RESULTS = Path("data/results")
COMBOS_FILE = RESULTS / "factor_combos_seedforge_031.csv"
SCREEN_FILE = RESULTS / "factor_screen_seedforge_031.csv"
SUMMARY_FILE = RESULTS / "summary_seedforge_032.csv"
PAIRED_FILE = RESULTS / "paired_comparison_seedforge_032.csv"
EQUITY_FILE = RESULTS / "equity_seedforge_032.csv"

TOP_COMBOS = 10
PORTFOLIO_SIZE = 50
INITIAL_CAPITAL = core.INITIAL_CAPITAL
POLICIES = ("rank_rebalance_only", "rank_rebalance_plus_rsi_partial")


@dataclass
class RunResult:
    equity: pd.Series
    exposure: pd.Series
    transactions: pd.DataFrame


def select_train_only_combos() -> pd.DataFrame:
    """Freeze finalists using only columns produced from 2014-2021."""
    if not COMBOS_FILE.exists() or not SCREEN_FILE.exists():
        raise FileNotFoundError(
            "SeedForge 031 결과가 없습니다. 먼저 python -u seedforge_031.py 를 실행하세요."
        )
    report = pd.read_csv(COMBOS_FILE)
    required = {"factor_combo", "train_stable", "search_score", "robust_ic"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"031 조합 CSV 필수 열이 없습니다: {sorted(missing)}")
    # Never use any opened_* column for ranking or filtering.
    return report.sort_values(
        ["train_stable", "search_score", "robust_ic"], ascending=False
    ).head(TOP_COMBOS).reset_index(drop=True)


def build_oriented_scores(
    combos: pd.DataFrame,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Rebuild only once, then retain matrices needed by frozen finalists."""
    directions = (
        pd.read_csv(SCREEN_FILE).set_index("factor")["direction"].astype(int).to_dict()
    )
    needed = {
        factor
        for text in combos["factor_combo"].astype(str)
        for factor in text.split(" + ")
    }
    flows = load_flow(find_flow_file(), close.columns)
    arrays, _groups = search.build_factor_library(close, open_, volume, dates, flows)
    missing = needed.difference(arrays)
    if missing:
        raise ValueError(f"031 최종 조합 팩터를 재구성하지 못했습니다: {sorted(missing)}")
    oriented: dict[str, np.ndarray] = {}
    for name in needed:
        if name not in directions:
            raise ValueError(f"031 단독 스크린에 방향이 없습니다: {name}")
        raw = arrays[name]
        oriented[name] = raw if directions[name] > 0 else 1 - raw
    return oriented


def target_schedule(
    combos: pd.DataFrame,
    oriented: dict[str, np.ndarray],
    signal_dates: pd.DatetimeIndex,
    market: core.SimulationInputs,
) -> dict[str, dict[int, tuple[int, ...]]]:
    """Map each month-end score to the next trading day's fixed top-N target."""
    schedules: dict[str, dict[int, tuple[int, ...]]] = {}
    for row in combos.itertuples(index=False):
        names = str(row.factor_combo).split(" + ")
        score = np.nanmean(np.stack([oriented[name] for name in names]), axis=0)
        schedule: dict[int, tuple[int, ...]] = {}
        for source_row, signal_date in enumerate(signal_dates):
            source_day = int(market.dates.searchsorted(signal_date, side="right") - 1)
            execution_day = source_day + 1
            if source_day < 0 or execution_day >= len(market.dates):
                continue
            values = score[source_row]
            valid = (
                np.isfinite(values)
                & market.entry_eligible[execution_day]
                & np.isfinite(market.open[execution_day])
                & (market.open[execution_day] > 0)
            )
            candidates = np.flatnonzero(valid)
            if not candidates.size:
                continue
            chosen = candidates[np.argsort(values[candidates])[-PORTFOLIO_SIZE:]]
            schedule[execution_day] = tuple(int(stock) for stock in chosen[::-1])
        schedules[str(row.factor_combo)] = schedule
    return schedules


def simulate_rank_portfolio(
    market: core.SimulationInputs,
    schedule: dict[int, tuple[int, ...]],
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    use_rsi: bool,
) -> RunResult:
    """Run a self-financing equal-weight portfolio with actual turnover costs."""
    cash = float(INITIAL_CAPITAL)
    shares: dict[int, float] = {}
    transactions: list[dict[str, object]] = []
    equity_values: list[float] = []
    exposure_values: list[float] = []

    def sell(day: int, stock: int, fraction: float, reason: str) -> None:
        nonlocal cash
        held = shares.get(stock, 0.0)
        price = market.open[day, stock]
        if held <= 0 or not (np.isfinite(price) and price > 0):
            return
        quantity = held * min(max(fraction, 0.0), 1.0)
        gross = quantity * price
        fee = gross * core.COST_PER_SIDE
        cash += gross - fee
        remaining = held - quantity
        if remaining <= held * 1e-10:
            shares.pop(stock, None)
        else:
            shares[stock] = remaining
        transactions.append({
            "date": market.dates[day], "ticker": market.tickers[stock],
            "side": "sell", "reason": reason, "gross": gross, "cost": fee,
        })

    def buy(day: int, stock: int, desired_gross: float) -> None:
        nonlocal cash
        price = market.open[day, stock]
        if desired_gross <= 0 or not market.entry_eligible[day, stock]:
            return
        if not (np.isfinite(price) and price > 0):
            return
        gross = min(desired_gross, cash / (1 + core.COST_PER_SIDE))
        if gross <= 0:
            return
        fee = gross * core.COST_PER_SIDE
        shares[stock] = shares.get(stock, 0.0) + gross / price
        cash -= gross + fee
        transactions.append({
            "date": market.dates[day], "ticker": market.tickers[stock],
            "side": "buy", "reason": "monthly_rank_rebalance", "gross": gross,
            "cost": fee,
        })

    for day in range(len(market.dates)):
        blocked_today: set[int] = set()
        if use_rsi and day > 0:
            # Confirmation at source-day close; execution is today's open.
            neutral_today = neutral.get(day - 1, set())
            overbought_today = overbought.get(day - 1, set()) - neutral_today
            for stock in tuple(shares):
                if stock in neutral_today:
                    sell(day, stock, 1.0, "rsi_neutral_full")
                    blocked_today.add(stock)
                elif stock in overbought_today:
                    sell(day, stock, 0.5, "rsi_overbought_half")
                    blocked_today.add(stock)

        if day in schedule:
            target = tuple(stock for stock in schedule[day] if stock not in blocked_today)
            target_set = set(target)
            for stock in tuple(shares):
                if stock not in target_set:
                    sell(day, stock, 1.0, "monthly_rank_exit")

            opening_equity = cash + sum(
                quantity * market.open[day, stock]
                for stock, quantity in shares.items()
                if np.isfinite(market.open[day, stock])
            )
            target_value = opening_equity / max(len(target), 1)
            # Sell overweight positions first so underweights use realized cash.
            for stock in target:
                held_value = shares.get(stock, 0.0) * market.open[day, stock]
                if np.isfinite(held_value) and held_value > target_value:
                    sell(day, stock, (held_value - target_value) / held_value, "monthly_reweight")
            for stock in target:
                held_value = shares.get(stock, 0.0) * market.open[day, stock]
                if np.isfinite(held_value) and held_value < target_value:
                    buy(day, stock, target_value - held_value)

        marked = sum(
            quantity * market.marked_close[day, stock]
            for stock, quantity in shares.items()
            if np.isfinite(market.marked_close[day, stock])
        )
        total = cash + marked
        equity_values.append(total)
        exposure_values.append(marked / total if total > 0 else 0.0)

    ledger = pd.DataFrame(transactions)
    return RunResult(
        pd.Series(equity_values, index=market.dates, name="equity"),
        pd.Series(exposure_values, index=market.dates, name="exposure"),
        ledger,
    )


def period_metric(equity: pd.Series, start: str | None, end: str | None) -> dict[str, float]:
    values = equity.loc[start:end].dropna() if start or end else equity.dropna()
    if len(values) < 2 or values.iloc[0] <= 0:
        return {"cagr": np.nan, "mdd": np.nan, "calmar": np.nan}
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
    mdd = float((values / values.cummax() - 1).min())
    return {"cagr": cagr, "mdd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan}


def summarize(
    combo: str,
    policy: str,
    frozen_rank: int,
    result: RunResult,
) -> dict[str, object]:
    row: dict[str, object] = {
        "build": BUILD, "frozen_train_rank": frozen_rank, "factor_combo": combo,
        "exit_policy": policy, "portfolio_size": PORTFOLIO_SIZE,
        "opened_used_for_selection": False, "passes_live_gate": False,
        "average_exposure": float(result.exposure.mean()),
        "transactions": len(result.transactions),
        "round_trip_equivalent_turnover": (
            float(result.transactions["gross"].sum()) / (2 * INITIAL_CAPITAL)
            if not result.transactions.empty else 0.0
        ),
        "total_transaction_cost": (
            float(result.transactions["cost"].sum()) if not result.transactions.empty else 0.0
        ),
    }
    for suffix, start, end in (
        ("full", None, None), ("train_2014_2021", "2014-01-01", "2021-12-31"),
        ("opened_2022_onward", "2022-01-01", None),
    ):
        row.update({f"{name}_{suffix}": value for name, value in period_metric(result.equity, start, end).items()})
    return row


def paired_report(summary: pd.DataFrame) -> pd.DataFrame:
    control = summary.loc[summary.exit_policy.eq(POLICIES[0])].set_index("factor_combo")
    overlay = summary.loc[summary.exit_policy.eq(POLICIES[1])].set_index("factor_combo")
    rows: list[dict[str, object]] = []
    for combo in control.index.intersection(overlay.index):
        left, right = control.loc[combo], overlay.loc[combo]
        row: dict[str, object] = {
            "factor_combo": combo, "frozen_train_rank": int(left.frozen_train_rank),
            "control_policy": POLICIES[0], "overlay_policy": POLICIES[1],
            "opened_used_for_selection": False, "passes_live_gate": False,
        }
        for period in ("full", "train_2014_2021", "opened_2022_onward"):
            row[f"delta_cagr_{period}"] = right[f"cagr_{period}"] - left[f"cagr_{period}"]
            row[f"mdd_improvement_{period}"] = right[f"mdd_{period}"] - left[f"mdd_{period}"]
            row[f"delta_calmar_{period}"] = right[f"calmar_{period}"] - left[f"calmar_{period}"]
        row["extra_transaction_cost"] = right.total_transaction_cost - left.total_transaction_cost
        row["extra_transactions"] = int(right.transactions - left.transactions)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["delta_calmar_train_2014_2021", "delta_cagr_train_2014_2021"], ascending=False
    )


def main() -> None:
    print(f"SeedForge 032 RSI 분할매도 paired test (build {BUILD})")
    combos = select_train_only_combos()
    market, _features, overbought, neutral, _kospi = core.load_or_prepare(False)
    close = pd.DataFrame(market.close, index=market.dates, columns=market.tickers)
    open_ = pd.DataFrame(market.open, index=market.dates, columns=market.tickers)
    volume = pd.DataFrame(market.volume, index=market.dates, columns=market.tickers)
    flows = load_flow(find_flow_file(), close.columns)
    signal_dates = pd.DatetimeIndex(flows["외국인"].index)
    oriented = build_oriented_scores(combos, close, open_, volume, signal_dates)
    schedules = target_schedule(combos, oriented, signal_dates, market)

    summaries: list[dict[str, object]] = []
    equity_columns: dict[str, pd.Series] = {}
    total = len(combos) * len(POLICIES)
    completed = 0
    for rank, row in enumerate(combos.itertuples(index=False), 1):
        combo = str(row.factor_combo)
        for policy in POLICIES:
            completed += 1
            result = simulate_rank_portfolio(
                market, schedules[combo], overbought, neutral,
                use_rsi=policy == POLICIES[1],
            )
            summary = summarize(combo, policy, rank, result)
            summaries.append(summary)
            equity_columns[f"rank{rank:02d}|{policy}|{combo}"] = result.equity
            print(
                f"[{completed:>2}/{total}] rank {rank:02d} {policy:<36} "
                f"train CAGR {summary['cagr_train_2014_2021']:.2%} "
                f"MDD {summary['mdd_train_2014_2021']:.1%}"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    paired = paired_report(summary_frame)
    summary_frame.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    paired.to_csv(PAIRED_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(equity_columns).to_csv(EQUITY_FILE, index_label="date", encoding="utf-8-sig")
    print("\nTrain 기준 RSI 분할매도 증분 결과")
    print(paired[[
        "frozen_train_rank", "factor_combo", "delta_cagr_train_2014_2021",
        "mdd_improvement_train_2014_2021", "delta_calmar_train_2014_2021",
        "extra_transaction_cost", "passes_live_gate",
    ]].to_string(index=False))
    print(f"\n요약 저장: {SUMMARY_FILE}")
    print(f"paired 저장: {PAIRED_FILE}")
    print(f"equity 저장: {EQUITY_FILE}")
    print("주의: 매수 목표·리밸런싱은 두 정책에서 동일하며 RSI 청산만 다릅니다.")
    print("주의: opened 결과는 진단일 뿐 선택 또는 실전 승인에 사용하지 않습니다.")


if __name__ == "__main__":
    main()
