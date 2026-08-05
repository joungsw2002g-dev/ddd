"""SeedForge 022: frozen-candidate forward-validation monitor.

This stage does not search for parameters.  It freezes the SeedForge 021
selection and measures only trades entered after the freeze date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
STATE_PATH = RESULTS / "forward_state_seedforge_022.json"
REPORT_PATH = RESULTS / "forward_report_seedforge_022.csv"


def selected_ledger(decision: pd.Series) -> Path:
    tag = (
        f"p{decision['selected_preset']}_x{decision['selected_exit_policy']}"
        f"_r{decision.get('selected_risk_policy', 'base')}"
    )
    return RESULTS / f"trades_{tag}.json"


def load_inputs() -> tuple[pd.Series, Path, pd.DataFrame]:
    decision_path = RESULTS / "decision_seedforge_021.csv"
    if not decision_path.exists():
        raise FileNotFoundError("decision_seedforge_021.csv가 없습니다. 먼저 SeedForge 021을 실행하세요.")
    decision = pd.read_csv(decision_path).iloc[0]
    ledger_path = selected_ledger(decision)
    if not ledger_path.exists():
        raise FileNotFoundError(f"최종 후보 거래원장이 없습니다: {ledger_path}")
    trades = pd.read_json(ledger_path)
    for column in ("entry_date", "exit_date"):
        trades[column] = pd.to_datetime(trades[column], errors="coerce")
    return decision, ledger_path, trades


def initialize(reset: bool) -> None:
    if STATE_PATH.exists() and not reset:
        raise FileExistsError("022 상태가 이미 있습니다. 다시 동결하려면 --initialize --reset을 사용하세요.")
    decision, ledger_path, trades = load_inputs()
    cutoff = trades["exit_date"].max()
    if pd.isna(cutoff):
        raise ValueError("거래원장에서 기준일을 계산할 수 없습니다.")
    state = {
        "version": 1,
        "cutoff_date": cutoff.date().isoformat(),
        "preset": decision["selected_preset"],
        "exit_policy": decision["selected_exit_policy"],
        "risk_policy": decision.get("selected_risk_policy", "base"),
        "ledger": ledger_path.name,
        "minimum_months": 6,
        "minimum_new_trades": 20,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SeedForge 022 동결 완료: {STATE_PATH}")
    print(f"전진검증 기준일: {state['cutoff_date']}")
    print("앞으로 데이터 갱신 후 021을 재실행하고, 이어서 python -u seedforge_022.py 를 실행하세요.")


def check() -> None:
    if not STATE_PATH.exists():
        raise FileNotFoundError("022 상태가 없습니다. python -u seedforge_022.py --initialize 를 먼저 실행하세요.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    decision, ledger_path, trades = load_inputs()
    identity = (decision["selected_preset"], decision["selected_exit_policy"], decision.get("selected_risk_policy", "base"))
    frozen = (state["preset"], state["exit_policy"], state["risk_policy"])
    if identity != frozen or ledger_path.name != state["ledger"]:
        raise ValueError("021 최종 후보가 동결 시점과 달라졌습니다. 파라미터를 원복하세요.")
    cutoff = pd.Timestamp(state["cutoff_date"])
    forward = trades.loc[trades["entry_date"] > cutoff].copy()
    returns = pd.to_numeric(forward.get("return", pd.Series(dtype=float)), errors="coerce").dropna()
    latest = trades["exit_date"].max()
    elapsed_months = max(0.0, (latest - cutoff).days / 30.4375) if pd.notna(latest) else 0.0
    row = {
        "cutoff_date": state["cutoff_date"],
        "latest_date": latest.date().isoformat() if pd.notna(latest) else "",
        "elapsed_months": elapsed_months,
        "new_trades": int(len(forward)),
        "avg_return": float(returns.mean()) if not returns.empty else float("nan"),
        "win_rate": float((returns > 0).mean()) if not returns.empty else float("nan"),
        "ready_for_review": elapsed_months >= state["minimum_months"] or len(forward) >= state["minimum_new_trades"],
    }
    pd.DataFrame([row]).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"전진검증 보고서 저장: {REPORT_PATH}")
    if not row["ready_for_review"]:
        print("아직 판정하지 않습니다. 6개월 또는 신규 거래 20건까지 계속 관찰하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SeedForge 022 frozen forward-validation monitor")
    parser.add_argument("--initialize", action="store_true", help="021 최종 후보와 기준일을 동결")
    parser.add_argument("--reset", action="store_true", help="기존 동결 상태를 명시적으로 교체")
    args = parser.parse_args()
    initialize(args.reset) if args.initialize else check()


if __name__ == "__main__":
    main()
