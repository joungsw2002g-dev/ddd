"""SeedForge 034: compare legacy 033 and audited risk engines on fixed samples.

This is an impact audit, not a new strategy search. Candidate selection only
uses train-era fields from the completed 033 summary. Legacy artifacts are
hashed before and after the run and are never written by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import seedforge_021 as core
import seedforge_033 as legacy
from seedforge_029 import find_flow_file, load_flow
from seedforge_risk_engine import (
    AssetStatus,
    CostModel,
    MarketFixture,
    PortfolioPolicy,
    WithdrawalRiskPolicy,
    simulate_audited_portfolio,
)


BUILD = "034-impact.0-20260820"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
LEGACY_SUMMARY = RESULTS / "summary_seedforge_033.csv"
LEGACY_FINALISTS = RESULTS / "finalists_seedforge_033.csv"
LEGACY_PROXY = RESULTS / "proxy_all_combos_seedforge_033.csv"
LEGACY_MANIFEST = RESULTS / "manifest_seedforge_033.json"
OUTPUT = RESULTS / "impact_sample_seedforge_034.csv"
MANIFEST_OUTPUT = RESULTS / "impact_manifest_seedforge_034.json"
ORDERS_OUTPUT = RESULTS / "impact_orders_seedforge_034.csv"
WITHDRAWALS_OUTPUT = RESULTS / "impact_withdrawals_seedforge_034.csv"
AUDIT_OUTPUT = RESULTS / "data_audit_seedforge_034.json"

LEGACY_FILES = (LEGACY_SUMMARY, LEGACY_FINALISTS, LEGACY_PROXY, LEGACY_MANIFEST)
FORBIDDEN_SELECTION_NAMES = ("opened", "test", "full")
REQUIRED_SUMMARY_COLUMNS = {
    "job_id", "factor_combo", "portfolio_size", "rebalance_months",
    "exit_rank_multiplier", "weight_policy", "train_selection_score", "turnover",
}

FIXED_CANDIDATES = (
    {
        "selection_reason": "fixed_value_representative",
        "factor_combo": "V02_value252", "portfolio_size": 80,
        "rebalance_months": 3, "exit_rank_multiplier": 2.0, "weight_policy": "equal",
    },
    {
        "selection_reason": "fixed_two_factor_representative",
        "factor_combo": "M03_ret120 + V02_value252", "portfolio_size": 50,
        "rebalance_months": 3, "exit_rank_multiplier": 2.0, "weight_policy": "equal",
    },
    {
        "selection_reason": "fixed_three_factor_representative",
        "factor_combo": "M03_ret120 + T02_ma60_distance + V02_value252",
        "portfolio_size": 20, "rebalance_months": 3,
        "exit_rank_multiplier": 2.0, "weight_policy": "discovery_ic",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="033 legacy vs audited engine impact sample")
    parser.add_argument("--audit-data", action="store_true")
    parser.add_argument("--allow-unresolved-status", action="store_true")
    parser.add_argument("--lifecycle-file", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=6, choices=range(3, 7))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def market_hash(market: core.SimulationInputs, fixture: MarketFixture) -> str:
    """Hash real market inputs in bounded row chunks without a full-size copy."""

    digest = hashlib.sha256()
    digest.update("\x01".join(map(str, market.dates)).encode())
    digest.update("\x02".join(map(str, market.tickers)).encode())
    arrays = (
        ("open", market.open), ("close", market.close),
        ("eligible", market.entry_eligible), ("status", fixture.status),
        ("recovery", fixture.delisting_recovery),
    )
    for name, source in arrays:
        array = np.asarray(source)
        digest.update(f"{name}:{array.shape}:{array.dtype.str}".encode())
        for start in range(0, len(array), 128):
            chunk = np.ascontiguousarray(array[start:start + 128])
            digest.update(chunk.tobytes())
    return digest.hexdigest()


def legacy_hashes() -> dict[str, str]:
    missing = [str(path) for path in LEGACY_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"033 필수 결과가 없습니다: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in LEGACY_FILES}


def read_summary_train_fields() -> pd.DataFrame:
    with LEGACY_SUMMARY.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = REQUIRED_SUMMARY_COLUMNS.difference(header)
        if missing:
            raise ValueError(f"033 summary 필수 열 누락: {sorted(missing)}")
        selected = sorted(REQUIRED_SUMMARY_COLUMNS)
        if any(term in name.lower() for name in selected for term in FORBIDDEN_SELECTION_NAMES):
            raise RuntimeError("후보 선택 열에 opened/test/full 열이 포함됐습니다.")
        records = [{name: row.get(name, "") for name in selected} for row in reader]
    frame = pd.DataFrame.from_records(records, columns=selected)
    if len(frame) != 108_000 or frame["job_id"].astype(str).nunique() != 108_000:
        raise RuntimeError(
            f"033 summary 무결성 실패: rows={len(frame):,}, "
            f"unique_jobs={frame['job_id'].astype(str).nunique():,}"
        )
    return frame


def _match_fixed(frame: pd.DataFrame, specification: dict[str, object]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for name, expected in specification.items():
        if name == "selection_reason":
            continue
        if isinstance(expected, float):
            mask &= np.isclose(pd.to_numeric(frame[name], errors="coerce"), expected)
        else:
            mask &= frame[name].astype(str).eq(str(expected))
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise RuntimeError(
            f"고정 후보는 정확히 1행이어야 합니다: {specification}, matches={len(matches)}"
        )
    row = matches.iloc[0].copy()
    row["selection_reason"] = specification["selection_reason"]
    return row


def select_fixed_candidates(frame: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Select fixed and train-only distribution samples without opened fields."""

    chosen = [_match_fixed(frame, specification) for specification in FIXED_CANDIDATES]
    used = {str(row["job_id"]) for row in chosen}

    score = pd.to_numeric(frame["train_selection_score"], errors="coerce")
    turnover = pd.to_numeric(frame["turnover"], errors="coerce")
    selectors = (
        ("train_score_median", score, 0.50),
        ("turnover_p01", turnover, 0.01),
        ("turnover_p99", turnover, 0.99),
    )
    for reason, values, quantile in selectors:
        if len(chosen) >= limit:
            break
        target = float(values.quantile(quantile))
        order = (values - target).abs().sort_values().index
        for index in order:
            row = frame.loc[index].copy()
            if str(row["job_id"]) in used:
                continue
            row["selection_reason"] = reason
            chosen.append(row)
            used.add(str(row["job_id"]))
            break
    if len(chosen) != limit:
        raise RuntimeError(f"대표 후보 선택 실패: expected={limit}, actual={len(chosen)}")
    return pd.DataFrame(chosen).reset_index(drop=True)


def find_lifecycle_file(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates: list[Path] = []
    for directory in (ROOT / "data" / "lifecycle", ROOT / "data" / "status"):
        if directory.exists():
            candidates.extend(directory.glob("*.parquet"))
            candidates.extend(directory.glob("*.csv"))
    return sorted(candidates)[0] if len(candidates) == 1 else None


def load_lifecycle(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    aliases = {
        "date": ("date", "날짜", "일자"),
        "ticker": ("ticker", "종목코드", "code"),
        "status": ("status", "거래상태", "상태"),
        "recovery": ("recovery_price", "회수가격", "정리매매가격"),
    }
    rename: dict[str, str] = {}
    for canonical, names in aliases.items():
        found = next((name for name in names if name in frame.columns), None)
        if found is not None:
            rename[found] = canonical
    frame = frame.rename(columns=rename)
    required = {"date", "ticker", "status"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"lifecycle 필수 열 누락: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("lifecycle date+ticker 중복")
    status_map = {
        "active": AssetStatus.ACTIVE, "정상": AssetStatus.ACTIVE,
        "trading": AssetStatus.ACTIVE, "suspended": AssetStatus.SUSPENDED,
        "거래정지": AssetStatus.SUSPENDED, "delisted": AssetStatus.DELISTED,
        "상장폐지": AssetStatus.DELISTED,
        "1": AssetStatus.ACTIVE, "2": AssetStatus.SUSPENDED, "3": AssetStatus.DELISTED,
    }
    mapped = frame["status"].astype(str).str.strip().str.lower().map(status_map)
    if mapped.isna().any():
        raise ValueError(f"알 수 없는 lifecycle status: {sorted(frame.loc[mapped.isna(), 'status'].unique())}")
    frame["status_code"] = mapped.astype(np.int8)
    if "recovery" not in frame:
        frame["recovery"] = np.nan
    frame["recovery"] = pd.to_numeric(frame["recovery"], errors="coerce")
    delisted = frame["status_code"].eq(AssetStatus.DELISTED)
    if frame.loc[delisted, "recovery"].isna().any() or (frame.loc[delisted, "recovery"] < 0).any():
        raise ValueError("DELISTED 행은 0 이상 recovery_price가 필요합니다.")
    return frame.sort_values(["ticker", "date"])


def build_market_fixture(
    market: core.SimulationInputs,
    lifecycle: pd.DataFrame | None,
    allow_unresolved: bool,
) -> tuple[MarketFixture, bool, dict[str, object]]:
    shape = market.open.shape
    status = np.where(
        np.isfinite(market.open) | np.isfinite(market.close),
        AssetStatus.ACTIVE,
        AssetStatus.SUSPENDED,
    ).astype(np.int8)
    recovery = np.full(shape, np.nan, dtype=float)
    status_complete = lifecycle is not None
    applied_events = 0
    if lifecycle is not None:
        ticker_map = {str(ticker).zfill(6): index for index, ticker in enumerate(market.tickers)}
        for ticker, events in lifecycle.groupby("ticker", sort=False):
            column = ticker_map.get(str(ticker).zfill(6))
            if column is None:
                continue
            for event in events.itertuples(index=False):
                day = int(market.dates.searchsorted(event.date, side="left"))
                if day >= len(market.dates):
                    continue
                code = int(event.status_code)
                status[day:, column] = code
                if code == AssetStatus.DELISTED:
                    recovery[day, column] = float(event.recovery)
                applied_events += 1
    elif not allow_unresolved:
        raise RuntimeError(
            "[DATA BLOCKED] Point-in-time lifecycle/status data가 없습니다. "
            "진단만 하려면 --allow-unresolved-status를 명시하세요."
        )
    fixture = MarketFixture(
        dates=pd.DatetimeIndex(market.dates), tickers=tuple(map(str, market.tickers)),
        open=np.asarray(market.open, dtype=float), close=np.asarray(market.close, dtype=float),
        eligible=np.asarray(market.entry_eligible, dtype=bool), status=status,
        delisting_recovery=recovery,
    )
    audit = {
        "status_data_complete": status_complete,
        "lifecycle_events_applied": applied_events,
        "active_cells": int(np.count_nonzero(status == AssetStatus.ACTIVE)),
        "suspended_cells": int(np.count_nonzero(status == AssetStatus.SUSPENDED)),
        "delisted_cells": int(np.count_nonzero(status == AssetStatus.DELISTED)),
    }
    return fixture, status_complete, audit


def metrics(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return {"cagr": math.nan, "mdd": math.nan}
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1)
    mdd = float((clean / clean.cummax() - 1).min())
    return {"cagr": cagr, "mdd": mdd}


def policy_variants() -> tuple[tuple[str, WithdrawalRiskPolicy | None], ...]:
    return (
        ("audited_base", None),
        ("audited_withdrawal", WithdrawalRiskPolicy(activate_aggression=False)),
        ("audited_withdrawal_aggressive", WithdrawalRiskPolicy()),
    )


def base_result_row(candidate: pd.Series, engine: str, risk_policy: str) -> dict[str, object]:
    return {
        "candidate_id": str(candidate["job_id"]), "job_id": str(candidate["job_id"]),
        "factor_combo": str(candidate["factor_combo"]),
        "selection_reason": str(candidate["selection_reason"]),
        "engine": engine, "risk_policy": risk_policy,
        "portfolio_size_before": int(candidate["portfolio_size"]),
        "rebalance_months_before": int(candidate["rebalance_months"]),
        "opened_used_for_selection": False, "passes_live_gate": False,
    }


def run_comparison(
    candidates: pd.DataFrame,
    singles: pd.DataFrame,
    oriented: dict[str, np.ndarray],
    signal_dates: pd.DatetimeIndex,
    market: core.SimulationInputs,
    fixture: MarketFixture,
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    status_complete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    orders: list[pd.DataFrame] = []
    withdrawals: list[pd.DataFrame] = []
    years = max((market.dates[-1] - market.dates[0]).days / 365.25, 1 / 365.25)

    for candidate in candidates.to_dict("records"):
        candidate = pd.Series(candidate)
        combo = tuple(str(candidate["factor_combo"]).split(" + "))
        config = legacy.PortfolioConfig(
            int(candidate["portfolio_size"]), int(candidate["rebalance_months"]),
            float(candidate["exit_rank_multiplier"]), str(candidate["weight_policy"]),
        )
        rankings = legacy.monthly_rankings(
            combo, config.weight_policy, singles, oriented, signal_dates, market,
            int(max(legacy.PORTFOLIO_SIZES) * max(legacy.EXIT_RANK_MULTIPLIERS)),
        )
        old = legacy.simulate_fixed_rsi(market, rankings, overbought, neutral, config)
        old_metrics = metrics(old.equity)
        legacy_row = base_result_row(candidate, "legacy_033", "legacy_no_withdrawal")
        legacy_row.update({
            "portfolio_size_after": config.size,
            "rebalance_months_after": config.rebalance_months,
            "final_account_nav": float(old.equity.iloc[-1]),
            "cumulative_withdrawals": 0.0, "final_total_wealth": float(old.equity.iloc[-1]),
            "account_cagr": old_metrics["cagr"], "total_wealth_cagr": old_metrics["cagr"],
            "account_mdd": old_metrics["mdd"], "total_wealth_mdd": old_metrics["mdd"],
            "transactions": old.transaction_count, "gross_traded": old.turnover * 2,
            "annual_turnover": (
                old.turnover * 2 / max(float(old.equity.mean()), 1e-15) / years
            ),
            "fees": old.transaction_cost, "taxes": math.nan, "slippage": math.nan,
            "forced_delisting_exits": math.nan, "unresolved_positions": math.nan,
            "withdrawal_count": 0, "first_withdrawal_date": "",
            "status_data_complete": status_complete, "passes_engine_audit": False,
        })
        rows.append(legacy_row)

        for name, withdrawal_policy in policy_variants():
            result = simulate_audited_portfolio(
                fixture, rankings, neutral, overbought,
                PortfolioPolicy(config.size, config.rebalance_months, config.exit_rank_multiplier),
                withdrawal=withdrawal_policy, costs=CostModel(),
            )
            daily = result.daily.set_index("date")
            account_metrics = metrics(daily["account_nav"])
            wealth_metrics = metrics(daily["total_wealth"])
            mean_nav = float(daily["account_nav"].mean())
            row = base_result_row(candidate, "audited_034", name)
            aggressive = withdrawal_policy is not None and withdrawal_policy.activate_aggression
            row.update({
                "portfolio_size_after": (
                    max(1, int(math.ceil(config.size * withdrawal_policy.aggressive_size_multiplier)))
                    if aggressive and not result.withdrawals.empty else config.size
                ),
                "rebalance_months_after": (
                    withdrawal_policy.aggressive_rebalance_months
                    if aggressive and not result.withdrawals.empty else config.rebalance_months
                ),
                "final_account_nav": result.final_account_nav,
                "cumulative_withdrawals": result.cumulative_withdrawals,
                "final_total_wealth": result.final_total_wealth,
                "account_cagr": account_metrics["cagr"],
                "total_wealth_cagr": wealth_metrics["cagr"],
                "account_mdd": account_metrics["mdd"],
                "total_wealth_mdd": wealth_metrics["mdd"],
                "transactions": int(len(result.ledger.loc[result.ledger["side"].isin(["buy", "sell"])])),
                "gross_traded": result.gross_traded,
                "annual_turnover": result.gross_traded / max(mean_nav, 1e-15) / years,
                "fees": result.fees, "taxes": result.taxes, "slippage": result.slippage,
                "forced_delisting_exits": result.forced_delisting_exits,
                "unresolved_positions": result.unresolved_positions,
                "withdrawal_count": len(result.withdrawals),
                "first_withdrawal_date": (
                    str(result.withdrawals.iloc[0]["date"]) if not result.withdrawals.empty else ""
                ),
                "status_data_complete": status_complete,
                "passes_engine_audit": bool(status_complete and result.unresolved_positions == 0),
            })
            rows.append(row)
            if not result.ledger.empty:
                ledger = result.ledger.copy()
                ledger.insert(0, "risk_policy", name)
                ledger.insert(0, "candidate_id", str(candidate["job_id"]))
                orders.append(ledger)
            if not result.withdrawals.empty:
                events = result.withdrawals.copy()
                events.insert(0, "risk_policy", name)
                events.insert(0, "candidate_id", str(candidate["job_id"]))
                withdrawals.append(events)
    return (
        pd.DataFrame(rows),
        pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(),
        pd.concat(withdrawals, ignore_index=True) if withdrawals else pd.DataFrame(),
    )


def main() -> int:
    args = parse_args()
    if any(path.exists() for path in (OUTPUT, MANIFEST_OUTPUT, ORDERS_OUTPUT, WITHDRAWALS_OUTPUT)):
        if not args.overwrite and not args.audit_data:
            raise FileExistsError("034 출력이 이미 있습니다. 교체하려면 --overwrite를 명시하세요.")

    before_hashes = legacy_hashes()
    summary = read_summary_train_fields()
    candidates = select_fixed_candidates(summary, args.candidate_limit)
    lifecycle_path = find_lifecycle_file(args.lifecycle_file)
    lifecycle = load_lifecycle(lifecycle_path) if lifecycle_path else None

    print("실시장 데이터 준비 중...", flush=True)
    market, _features, overbought, neutral, _kospi = core.load_or_prepare(False)
    fixture, status_complete, data_audit = build_market_fixture(
        market, lifecycle, args.allow_unresolved_status or args.audit_data
    )
    data_audit.update({
        "build": BUILD, "lifecycle_file": str(lifecycle_path) if lifecycle_path else "",
        "allow_unresolved_status": args.allow_unresolved_status,
        "market_rows": len(market.dates), "market_tickers": len(market.tickers),
        "candidate_count": len(candidates), "legacy_hashes": before_hashes,
        "readiness": "READY" if status_complete else "DATA_BLOCKED_LIFECYCLE",
    })
    atomic_json(AUDIT_OUTPUT, data_audit)
    print(json.dumps(data_audit, ensure_ascii=False, indent=2))
    if args.audit_data:
        if not status_complete:
            print("[DATA BLOCKED] authoritative lifecycle/status/recovery data가 없습니다.")
        print(f"[AUDIT ONLY] {AUDIT_OUTPUT}")
        return 0

    flow_path = find_flow_file()
    flows = load_flow(flow_path, pd.Index(market.tickers))
    signal_dates = pd.DatetimeIndex(flows["외국인"].index)
    close = pd.DataFrame(market.close, index=market.dates, columns=market.tickers)
    open_ = pd.DataFrame(market.open, index=market.dates, columns=market.tickers)
    volume = pd.DataFrame(market.volume, index=market.dates, columns=market.tickers)
    singles = legacy.load_single_universe(False)
    oriented, _groups = legacy.build_oriented_library(
        singles, close, open_, volume, signal_dates
    )
    results, orders, withdrawal_events = run_comparison(
        candidates, singles, oriented, signal_dates, market, fixture,
        overbought, neutral, status_complete,
    )
    expected = len(candidates) * 4
    if len(results) != expected:
        raise RuntimeError(f"034 결과 행 수 불일치: expected={expected}, actual={len(results)}")
    if results["opened_used_for_selection"].any() or results["passes_live_gate"].any():
        raise RuntimeError("opened/live guard violation")

    after_hashes = legacy_hashes()
    if before_hashes != after_hashes:
        raise RuntimeError("033 legacy 파일이 실행 중 변경됐습니다. 034 출력을 공개하지 않습니다.")
    atomic_csv(OUTPUT, results)
    atomic_csv(ORDERS_OUTPUT, orders)
    atomic_csv(WITHDRAWALS_OUTPUT, withdrawal_events)
    script_paths = (ROOT / "seedforge_021.py", ROOT / "seedforge_033.py",
                    ROOT / "seedforge_risk_engine.py", Path(__file__).resolve())
    factor_paths = (
        ROOT / "seedforge_025_screen.py", ROOT / "seedforge_029.py",
        ROOT / "seedforge_031.py",
    )
    source_paths = [Path(flow_path), ROOT / "data" / "dart" / "financials.parquet"]
    input_data_hashes = {
        str(path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path): sha256_file(path)
        for path in source_paths if path.exists()
    }
    manifest = {
        "build": BUILD, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "legacy_hashes": before_hashes,
        "code_hashes": {path.name: sha256_file(path) for path in script_paths},
        "factor_definition_hashes": {path.name: sha256_file(path) for path in factor_paths},
        "market_data_hash": market_hash(market, fixture),
        "input_data_hashes": input_data_hashes,
        "status_data_complete": status_complete,
        "lifecycle_file": str(lifecycle_path) if lifecycle_path else "",
        "candidate_selection_fields": sorted(REQUIRED_SUMMARY_COLUMNS),
        "candidate_job_ids": candidates["job_id"].astype(str).tolist(),
        "candidate_selection_reasons": candidates["selection_reason"].tolist(),
        "policies": ["legacy_no_withdrawal", *[name for name, _ in policy_variants()]],
        "cost_model": asdict(CostModel()),
        "withdrawal_policy": asdict(WithdrawalRiskPolicy()),
        "withdrawn_cash_return_assumption": 0.0,
        "opened_used_for_selection": False, "passes_live_gate": False,
        "result_rows": len(results), "order_rows": len(orders),
        "withdrawal_rows": len(withdrawal_events),
    }
    atomic_json(MANIFEST_OUTPUT, manifest)
    print(f"[DONE] 비교 {len(results)}행: {OUTPUT}")
    print(f"주문 원장: {ORDERS_OUTPUT}")
    print(f"출금 원장: {WITHDRAWALS_OUTPUT}")
    print(f"manifest: {MANIFEST_OUTPUT}")
    print("주의: passes_live_gate=False이며 실전 승인이 아닙니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
