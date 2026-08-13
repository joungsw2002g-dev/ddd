"""SeedForge 연구실 1차 버전: 한국어 로컬 연구 대시보드.

외부 웹 서버나 계정 없이 ``C:\alpha`` 안에서만 실행한다. 현재 버전은
SeedForge 033 대규모 매수 탐색을 화면에서 시작/중단/재개하고, 누적 결과를
읽어 비교하며, 이후 손익비·RSI·순위 청산 비교 계획을 JSON으로 보존한다.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BUILD = "studio-1.0-20260813"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
STATE_FILE = RESULTS / "seedforge_studio_state.json"
LOG_FILE = RESULTS / "seedforge_studio_current.log"
SUMMARY_033 = RESULTS / "summary_seedforge_033.csv"
MANIFEST_033 = RESULTS / "manifest_seedforge_033.json"
HOST, PORT = "127.0.0.1", 8765

PROCESS: subprocess.Popen[str] | None = None
PROCESS_LOCK = threading.Lock()
STARTED_AT: float | None = None


FACTOR_GROUPS = [
    ("가치", "가격이 과거 또는 재무가치 대비 낮은 종목"),
    ("모멘텀", "최근 수익 흐름이 강하거나 여러 기간의 방향이 일치하는 종목"),
    ("추세", "이동평균과 현재 가격의 위치로 상승·하락 흐름을 측정"),
    ("저변동성·위험", "가격 흔들림, 왜도, 낙폭이 상대적으로 낮은 종목"),
    ("가격행동", "시가·종가·갭 등 하루 가격 움직임의 특성"),
    ("보조지표", "RSI·코폭 등 가격에서 계산한 과열·순환 지표"),
    ("거래량", "거래량과 거래대금의 변화 및 가격 대비 수급 강도"),
    ("투자자 수급", "외국인·기관·개인의 월별 순매수 흐름"),
    ("DART 재무", "공시 접수일 이후에만 사용하는 수익성·현금흐름·성장성"),
]

SELL_RULES = [
    ("순위 청산", "보유 종목이 정한 이탈 순위 밖으로 밀리면 다음 리밸런싱에 매도"),
    ("RSI 분할매도", "중립 약세 다이버전스는 전량, 과매수 약세 다이버전스는 50% 매도"),
    ("RSI 지연 본절", "과매수 분할매도 후 정해진 횟수부터 잔량 본절 청산을 활성화"),
    ("고정 손익비", "손절폭을 1위험(R)으로 두고 1R·1.5R·2R·3R·4R에서 익절"),
    ("고정 손절만", "익절 목표 없이 손절과 최대 보유기간만 적용"),
    ("변동성 손익비", "종목 변동성에 따라 손절폭을 다르게 하고 익절은 R배수로 결정"),
    ("최대 보유기간", "20·60·120·252거래일이 지나면 조건과 무관하게 청산"),
]

RISK_RULES = [
    "기본", "중립 첫 신호 70% 매도", "중립 첫 신호 50% 매도",
    "손실 중립 신호 절반 매도", "2년 보유 손실 청산", "252일 후 -20% 청산",
    "252일 후 -25% 청산", "KOSPI 약세 중 손실 청산", "3년 보유 청산",
    "2년 보유·KOSPI 약세 청산", "고점수익 반납 방지", "추적손절",
]

GLOSSARY = [
    ("팩터", "종목을 비교·정렬하는 수치. 화면에서는 ‘선정 기준’으로 표시합니다."),
    ("프록시", "빠른 1차 예비평가. 실제 포트폴리오 전체 계산 전에 후보를 확인합니다."),
    ("Discovery", "2014~2017 탐색 구간. 화면에서는 ‘발견 구간’으로 표시합니다."),
    ("Validation", "2018~2021 확인 구간. 화면에서는 ‘검증 구간’으로 표시합니다."),
    ("Opened", "이미 여러 번 본 2022년 이후 진단 구간. 후보 선택에는 사용할 수 없습니다."),
    ("CAGR", "연복리수익률. 전체 기간 수익을 1년 기준 복리수익으로 환산한 값입니다."),
    ("MDD", "최대낙폭. 자산 최고점에서 최저점까지 가장 크게 떨어진 비율입니다."),
    ("Calmar", "연복리수익률을 최대낙폭으로 나눈 위험 대비 수익 지표입니다."),
    ("IC", "선정 기준 순위와 이후 수익률 순위가 얼마나 같은 방향인지 나타냅니다."),
    ("회전율", "포트폴리오가 얼마나 자주 교체되는지 나타내며 높을수록 비용 부담이 큽니다."),
    ("R", "한 거래에서 감수하는 손절폭 1단위. 손절 -10%라면 2R 목표는 +20%입니다."),
    ("다이버전스", "가격 고점은 높아지지만 RSI 고점은 낮아지는 약세 신호입니다."),
]


def default_state() -> dict[str, object]:
    return {
        "build": BUILD,
        "buy": {
            "max_factors": 4, "same_group": True, "include_unstable": True,
            "full_finalists": 0, "max_jobs": 5000,
            "factor_groups": [name for name, _ in FACTOR_GROUPS],
        },
        "portfolio": {
            "sizes": [20, 30, 50, 80], "rebalance_months": [1, 2, 3],
            "exit_rank_multipliers": [1.0, 1.5, 2.0],
            "weights": ["동일가중", "발견구간 IC", "검증구간 IC"],
            "roundtrip_costs": [0.0058, 0.01, 0.015],
        },
        "sell_comparison_plan": {
            "rules": [name for name, _ in SELL_RULES],
            "stop_losses": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
            "reward_multiples": [1.0, 1.5, 2.0, 3.0, 4.0],
            "max_holding_days": [20, 60, 120, 252],
        },
        "validation": {
            "opened_locked": True, "cost_stress": True,
            "concentration_exclusion": True, "forward_months": 6,
            "forward_trades": 20,
        },
    }


def load_state() -> dict[str, object]:
    if not STATE_FILE.exists():
        return default_state()
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()


def save_state(state: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    state["build"] = BUILD
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def process_status() -> dict[str, object]:
    global PROCESS
    with PROCESS_LOCK:
        running = PROCESS is not None and PROCESS.poll() is None
        code = None if PROCESS is None or running else PROCESS.returncode
    manifest: dict[str, object] = {}
    if MANIFEST_033.exists():
        try:
            manifest = json.loads(MANIFEST_033.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    completed = 0
    if SUMMARY_033.exists():
        try:
            with SUMMARY_033.open("r", encoding="utf-8-sig", newline="") as file:
                completed = max(sum(1 for _ in file) - 1, 0)
        except OSError:
            pass
    total = int(manifest.get("full_portfolio_jobs", 0) or 0)
    return {
        "running": running, "exit_code": code, "completed": completed, "total": total,
        "percent": round(completed / total * 100, 2) if total else 0,
        "started_at": STARTED_AT, "manifest": manifest,
    }


def tail_log(lines: int = 100) -> str:
    if not LOG_FILE.exists():
        return "아직 실행 기록이 없습니다."
    try:
        content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError as exc:
        return f"기록을 읽지 못했습니다: {exc}"


def top_results(limit: int = 30) -> dict[str, object]:
    if not SUMMARY_033.exists():
        return {"columns": [], "rows": [], "message": "033 결과 파일이 아직 없습니다."}
    try:
        with SUMMARY_033.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError as exc:
        return {"columns": [], "rows": [], "message": str(exc)}
    numeric = ["train_selection_score", "robust_train_cagr", "cagr_validation"]
    def key(row: dict[str, str]) -> tuple[float, ...]:
        values = []
        for column in numeric:
            try:
                values.append(float(row.get(column, "-inf")))
            except ValueError:
                values.append(float("-inf"))
        return tuple(values)
    rows.sort(key=key, reverse=True)
    columns = [
        "factor_combo", "portfolio_size", "rebalance_months", "exit_rank_multiplier",
        "weight_policy", "cagr_discovery", "cagr_validation", "mdd_train",
        "calmar_train", "robust_train_cagr", "turnover", "cagr_opened",
    ]
    return {"columns": columns, "rows": [{c: r.get(c, "") for c in columns} for r in rows[:limit]]}


def validate_start(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key, low, high in (("max_factors", 1, 6), ("full_finalists", 0, 1_000_000), ("max_jobs", 0, 1_000_000)):
        try:
            value = int(payload.get(key, 0))
        except (TypeError, ValueError):
            errors.append(f"{key} 값이 숫자가 아닙니다.")
            continue
        if not low <= value <= high:
            errors.append(f"{key} 값은 {low}~{high} 범위여야 합니다.")
    if not (ROOT / "seedforge_033.py").exists():
        errors.append("같은 폴더에 seedforge_033.py가 없습니다.")
    return errors


def start_033(payload: dict[str, object]) -> tuple[bool, str]:
    global PROCESS, STARTED_AT
    errors = validate_start(payload)
    if errors:
        return False, " ".join(errors)
    with PROCESS_LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            return False, "이미 실험이 실행 중입니다."
        command = [sys.executable, "-u", "seedforge_033.py", "--max-factors", str(int(payload["max_factors"]))]
        if bool(payload.get("same_group")):
            command.append("--same-group")
        if bool(payload.get("include_unstable")):
            command.append("--include-unstable-singles")
        command.extend(["--full-finalists", str(int(payload["full_finalists"]))])
        if int(payload["max_jobs"]):
            command.extend(["--max-jobs", str(int(payload["max_jobs"]))])
        if bool(payload.get("restart")):
            command.append("--restart")
        RESULTS.mkdir(parents=True, exist_ok=True)
        log = LOG_FILE.open("a", encoding="utf-8")
        log.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 실행: {' '.join(command)}\n")
        log.flush()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        PROCESS = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, creationflags=flags,
        )
        STARTED_AT = time.time()
    return True, "SeedForge 033을 시작했습니다. 화면을 닫아도 계산은 계속됩니다."


def stop_process() -> tuple[bool, str]:
    global PROCESS
    with PROCESS_LOCK:
        if PROCESS is None or PROCESS.poll() is not None:
            return False, "실행 중인 작업이 없습니다."
        if os.name == "nt":
            PROCESS.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            PROCESS.terminate()
    return True, "중단 신호를 보냈습니다. 저장된 지점부터 다시 실행할 수 있습니다."


def options_html(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<label class="option"><input type="checkbox" checked><span><b>{name}</b><small>{desc}</small></span></label>'
        for name, desc in items
    )


HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeedForge 연구실</title><style>
:root{--bg:#0b0f14;--panel:#121821;--panel2:#17202b;--line:#273241;--text:#e9f0f7;--muted:#91a0b3;--blue:#4aa3ff;--cyan:#43d9c3;--green:#64d58a;--yellow:#f2c14e;--red:#ff6b78;--shadow:0 18px 45px #0007}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -10%,#17314c 0,transparent 30%),var(--bg);color:var(--text);font:14px/1.55 "Pretendard","Noto Sans KR","Malgun Gothic",sans-serif}
button,input,select{font:inherit}header{height:68px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 30px;position:sticky;top:0;background:#0b0f14ed;backdrop-filter:blur(14px);z-index:5}.brand{display:flex;align-items:center;gap:12px}.logo{width:36px;height:36px;border-radius:11px;background:linear-gradient(135deg,var(--blue),var(--cyan));box-shadow:0 0 28px #4aa3ff55;display:grid;place-items:center;color:#071018;font-weight:900}.brand h1{font-size:18px;margin:0}.brand p{margin:0;color:var(--muted);font-size:12px}.badge{padding:6px 10px;border:1px solid #28587b;background:#102536;color:#8bcaff;border-radius:999px;font-size:12px}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:calc(100vh - 68px)}nav{padding:22px 14px;border-right:1px solid var(--line);background:#0e131a}nav button{width:100%;border:0;background:transparent;color:var(--muted);padding:11px 13px;text-align:left;border-radius:9px;margin-bottom:4px;cursor:pointer}nav button:hover,nav button.active{background:#172332;color:#fff}nav .section{color:#56667a;font-size:11px;font-weight:700;padding:15px 13px 7px;letter-spacing:.08em}main{padding:26px;max-width:1500px;width:100%;margin:auto}.view{display:none}.view.active{display:block}.title-row{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:20px}.title-row h2{font-size:25px;margin:0 0 4px}.title-row p{color:var(--muted);margin:0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:linear-gradient(145deg,#141b24,#10161e);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.card{padding:18px}.card label{color:var(--muted);font-size:12px}.metric{font-size:25px;font-weight:800;margin-top:5px}.metric.blue{color:#83c4ff}.metric.green{color:var(--green)}.metric.yellow{color:var(--yellow)}.panel{padding:20px;margin-top:16px}.panel h3{margin:0 0 4px;font-size:16px}.sub{color:var(--muted);font-size:12px;margin-bottom:16px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.option{display:flex;gap:10px;padding:12px;border:1px solid var(--line);background:#101720;border-radius:10px}.option input{accent-color:var(--blue);margin-top:3px}.option span{display:flex;flex-direction:column}.option small{color:var(--muted);margin-top:2px}.fields{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.field label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.field input,.field select{width:100%;background:#0c1219;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px}.actions{display:flex;gap:9px;margin-top:18px}.btn{border:0;border-radius:9px;padding:10px 15px;font-weight:700;cursor:pointer}.primary{background:linear-gradient(135deg,#2389ee,#39b9d4);color:white}.secondary{background:#202b38;color:#dbe8f5;border:1px solid #314154}.danger{background:#3a1b22;color:#ffacb4;border:1px solid #65303b}.warning{background:#2e2817;color:#f2d37e;border:1px solid #594b21}.progress{height:10px;background:#091018;border-radius:99px;overflow:hidden;border:1px solid #253344}.progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--cyan));transition:.4s}.statusline{display:flex;justify-content:space-between;margin:8px 0;color:var(--muted)}pre{background:#080d12;border:1px solid #202b37;border-radius:10px;padding:14px;max-height:310px;overflow:auto;color:#b9c9d8;font:12px/1.5 Consolas,monospace;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:12px}th{position:sticky;top:68px;background:#17212d;color:#a9bad0;text-align:left;padding:10px;border-bottom:1px solid #334253}td{padding:9px 10px;border-bottom:1px solid #202a36;max-width:330px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}tr:hover td{background:#15202b}.table-wrap{overflow:auto;max-height:600px;border:1px solid var(--line);border-radius:10px}.note{border-left:3px solid var(--yellow);background:#242013;padding:12px 14px;border-radius:6px;color:#e4d6a4}.glossary{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.term{padding:13px;border:1px solid var(--line);border-radius:10px;background:#101720}.term b{color:#8cc8ff}.term p{margin:4px 0 0;color:var(--muted)}.tag{display:inline-block;padding:3px 7px;border-radius:6px;background:#1d2b39;color:#99cfff;margin:2px;font-size:11px}.toast{position:fixed;right:24px;bottom:24px;padding:13px 16px;background:#172331;border:1px solid #36516c;border-radius:10px;box-shadow:var(--shadow);display:none;max-width:420px}.toast.show{display:block}@media(max-width:950px){.layout{grid-template-columns:1fr}nav{display:none}.cards,.grid2,.grid3,.fields,.glossary{grid-template-columns:1fr 1fr}main{padding:16px}}@media(max-width:620px){.cards,.grid2,.grid3,.fields,.glossary{grid-template-columns:1fr}}
</style></head><body>
<header><div class="brand"><div class="logo">SF</div><div><h1>SeedForge 연구실</h1><p>한국 주식 전략 연구 · 로컬 전용</p></div></div><span class="badge">열린 구간 선택 잠금</span></header>
<div class="layout"><nav><div class="section">연구</div><button class="active" data-view="home">⌂ 연구 현황</button><button data-view="buy">◎ 매수 조건</button><button data-view="sell">⇄ 매도·위험관리</button><button data-view="run">▶ 실험 실행</button><div class="section">분석</div><button data-view="results">▤ 결과 비교</button><button data-view="terms">? 용어 설명</button></nav><main>
<section id="home" class="view active"><div class="title-row"><div><h2>연구 현황</h2><p>긴 실험의 진행률과 현재 데이터 상태를 한눈에 확인합니다.</p></div><button class="btn secondary" onclick="refreshAll()">새로고침</button></div>
<div class="cards"><div class="card"><label>실행 상태</label><div id="runState" class="metric blue">확인 중</div></div><div class="card"><label>완료 작업</label><div id="doneJobs" class="metric">0</div></div><div class="card"><label>전체 작업</label><div id="totalJobs" class="metric">0</div></div><div class="card"><label>진행률</label><div id="percent" class="metric green">0%</div></div></div>
<div class="panel"><h3>현재 실험 진행</h3><div class="statusline"><span id="progressText">실행 정보를 기다리는 중</span><span id="progressPct">0%</span></div><div class="progress"><i id="bar"></i></div><div class="actions"><button class="btn primary" onclick="showView('run')">실험 설정으로 이동</button><button class="btn danger" onclick="stopRun()">안전하게 중단</button></div></div>
<div class="grid2"><div class="panel"><h3>데이터 준비 상태</h3><div class="sub">저장소에서 확인된 자료</div><p>● OHLCV · KOSPI · DART 재무 · 투자자 수급</p><p style="color:var(--red)">● 시가총액 이력 · 업종 이력은 없음</p><div class="note">없는 자료가 필요한 기준은 자동으로 제외하며 가짜 값으로 채우지 않습니다.</div></div><div class="panel"><h3>최근 실행 기록</h3><pre id="log">불러오는 중...</pre></div></div></section>

<section id="buy" class="view"><div class="title-row"><div><h2>매수 조건</h2><p>저장소에 축적된 선정 기준을 한국어 설명과 함께 구성합니다.</p></div></div><div class="panel"><h3>선정 기준 계열</h3><div class="sub">현재 033 엔진은 실제 파일에 존재하는 세부 기준만 계산합니다.</div><div class="grid3">__FACTOR_OPTIONS__</div></div><div class="panel"><h3>조합 범위</h3><div class="fields"><div class="field"><label>한 조합의 최대 기준 수</label><select id="maxFactors"><option>2</option><option>3</option><option selected>4</option><option>5</option><option>6</option></select></div><div class="field"><label>실제 정밀평가 조합 수 (0=전부)</label><input id="fullFinalists" type="number" min="0" value="0"></div><div class="field"><label>한 번에 실행할 작업 수 (0=제한 없음)</label><input id="maxJobs" type="number" min="0" value="5000"></div></div><div class="actions"><label class="option"><input id="sameGroup" type="checkbox" checked><span><b>같은 계열끼리 조합 허용</b><small>경우의 수가 크게 늘어납니다.</small></span></label><label class="option"><input id="unstable" type="checkbox" checked><span><b>단독 안정성 미통과 포함</b><small>누락은 줄지만 과최적화 위험이 커집니다.</small></span></label></div></div></section>

<section id="sell" class="view"><div class="title-row"><div><h2>매도·위험관리</h2><p>과거 대화와 저장소에 남아 있는 청산 아이디어를 비교 계획으로 보존합니다.</p></div></div><div class="panel"><h3>매도 방식</h3><div class="grid2">__SELL_OPTIONS__</div></div><div class="panel"><h3>손익비 후보</h3><div class="sub">현재 실행 중인 033은 RSI 고정입니다. 아래 값은 033 상위 매수식에 적용할 다음 비교 실험 계획입니다.</div><div class="fields"><div class="field"><label>손절폭</label><input value="5%, 8%, 10%, 12%, 15%, 20%"></div><div class="field"><label>목표 손익비</label><input value="1R, 1.5R, 2R, 3R, 4R"></div><div class="field"><label>최대 보유기간</label><input value="20, 60, 120, 252 거래일"></div></div></div><div class="panel"><h3>저장소의 기존 위험관리 후보</h3><div>__RISK_TAGS__</div><div class="note" style="margin-top:14px">모든 정책을 한꺼번에 최적화하지 않습니다. 033 매수 후보를 고정한 뒤 동일 매수식에 대해 짝지어 비교합니다.</div></div></section>

<section id="run" class="view"><div class="title-row"><div><h2>실험 실행</h2><p>CMD 명령 대신 화면에서 SeedForge 033을 시작하고 저장 지점부터 이어갑니다.</p></div></div><div class="panel"><h3>실행 설정 확인</h3><div class="sub">033은 RSI 분할매도를 고정하고 매수 조합 및 포트폴리오 구성만 탐색합니다.</div><div class="grid2"><div><p><b>보유 종목 수</b></p><span class="tag">20</span><span class="tag">30</span><span class="tag">50</span><span class="tag">80</span><p><b>리밸런싱</b></p><span class="tag">매월</span><span class="tag">2개월</span><span class="tag">분기</span></div><div><p><b>이탈 순위 완충</b></p><span class="tag">1배</span><span class="tag">1.5배</span><span class="tag">2배</span><p><b>가중 방식</b></p><span class="tag">동일가중</span><span class="tag">발견구간 IC</span><span class="tag">검증구간 IC</span></div></div><div class="actions"><button class="btn primary" onclick="startRun(false)">저장 지점부터 시작</button><button class="btn warning" onclick="startRun(true)">결과 삭제 후 처음부터</button><button class="btn danger" onclick="stopRun()">중단</button></div><div class="note" style="margin-top:15px">‘처음부터’는 033 결과를 삭제합니다. 현재 실행을 이어갈 때는 반드시 ‘저장 지점부터 시작’을 사용하세요.</div></div></section>

<section id="results" class="view"><div class="title-row"><div><h2>결과 비교</h2><p>후보 선택은 발견·검증 구간으로만 하며 열린 구간은 참고 열입니다.</p></div><button class="btn secondary" onclick="loadResults()">결과 새로고침</button></div><div class="panel"><div class="table-wrap"><table><thead id="resultHead"></thead><tbody id="resultBody"></tbody></table></div></div></section>

<section id="terms" class="view"><div class="title-row"><div><h2>용어 설명</h2><p>영문 전문용어는 한국에서 흔히 쓰는 표현과 짧은 설명을 함께 표시합니다.</p></div></div><div class="glossary">__GLOSSARY__</div></section>
</main></div><div id="toast" class="toast"></div>
<script>
const labels={factor_combo:'매수 기준 조합',portfolio_size:'보유 종목',rebalance_months:'리밸런싱(개월)',exit_rank_multiplier:'이탈 완충',weight_policy:'가중 방식',cagr_discovery:'발견 연복리',cagr_validation:'검증 연복리',mdd_train:'훈련 최대낙폭',calmar_train:'훈련 칼마',robust_train_cagr:'보수 연복리',turnover:'회전율',cagr_opened:'열린 구간 연복리'};
document.querySelectorAll('nav button[data-view]').forEach(b=>b.onclick=()=>showView(b.dataset.view));
function showView(id){document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x.dataset.view===id));if(id==='results')loadResults()}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),4500)}
async function api(path,options){const r=await fetch(path,options);return await r.json()}
async function refreshAll(){const s=await api('/api/status');document.getElementById('runState').textContent=s.running?'실행 중':(s.exit_code===0?'완료':'대기');document.getElementById('doneJobs').textContent=s.completed.toLocaleString();document.getElementById('totalJobs').textContent=s.total.toLocaleString();document.getElementById('percent').textContent=s.percent+'%';document.getElementById('progressPct').textContent=s.percent+'%';document.getElementById('bar').style.width=s.percent+'%';document.getElementById('progressText').textContent=s.total?`${s.completed.toLocaleString()} / ${s.total.toLocaleString()} 작업`:'실행 원장 대기 중';const l=await api('/api/log');document.getElementById('log').textContent=l.log}
async function startRun(restart){const body={max_factors:+maxFactors.value,full_finalists:+fullFinalists.value,max_jobs:+maxJobs.value,same_group:sameGroup.checked,include_unstable:unstable.checked,restart};const r=await api('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});toast(r.message);refreshAll()}
async function stopRun(){const r=await api('/api/stop',{method:'POST'});toast(r.message);refreshAll()}
function fmt(v,c){if(v===''||v==null)return '';if(c.includes('cagr')||c.includes('mdd')){const n=Number(v);return isNaN(n)?v:(n*100).toFixed(2)+'%'}if(['calmar_train','turnover'].includes(c)){const n=Number(v);return isNaN(n)?v:n.toFixed(3)}return v}
async function loadResults(){const d=await api('/api/results');resultHead.innerHTML='<tr>'+d.columns.map(c=>`<th title="${c}">${labels[c]||c}</th>`).join('')+'</tr>';resultBody.innerHTML=d.rows.map(r=>'<tr>'+d.columns.map(c=>`<td title="${r[c]||''}">${fmt(r[c],c)}</td>`).join('')+'</tr>').join('');if(d.message)toast(d.message)}
refreshAll();setInterval(refreshAll,5000);
</script></body></html>'''


def render_html() -> str:
    glossary = "".join(f'<div class="term"><b>{term}</b><p>{desc}</p></div>' for term, desc in GLOSSARY)
    risks = "".join(f'<span class="tag">{name}</span>' for name in RISK_RULES)
    return (HTML.replace("__FACTOR_OPTIONS__", options_html(FACTOR_GROUPS))
            .replace("__SELL_OPTIONS__", options_html(SELL_RULES))
            .replace("__RISK_TAGS__", risks).replace("__GLOSSARY__", glossary))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(render_html().encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send_json(process_status())
        elif path == "/api/log":
            self.send_json({"log": tail_log()})
        elif path == "/api/results":
            self.send_json(top_results())
        elif path == "/api/state":
            self.send_json(load_state())
        else:
            self.send_json({"message": "경로를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("요청이 너무 큽니다.")
        return json.loads(self.rfile.read(length).decode() or "{}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"ok": False, "message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/start":
            ok, message = start_033(payload)
            self.send_json({"ok": ok, "message": message}, 200 if ok else 409)
        elif path == "/api/stop":
            ok, message = stop_process()
            self.send_json({"ok": ok, "message": message}, 200 if ok else 409)
        elif path == "/api/state":
            save_state(payload)
            self.send_json({"ok": True, "message": "연구 설정을 저장했습니다."})
        else:
            self.send_json({"message": "경로를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        save_state(default_state())
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"SeedForge 연구실 {BUILD}")
    print(f"브라우저 주소: {url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n연구실 화면을 종료합니다. 별도 실행된 연구 작업은 저장 지점까지 유지됩니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
