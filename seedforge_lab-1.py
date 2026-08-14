"""SeedForge 무코드 전략 실험실 1차 버전.

저장소에 축적된 기술·수급·DART 지표를 자동으로 불러와 매수 신호,
매도 신호, 매수/매도 결합의 세 모드로 월별 사건연구를 수행한다.
Tkinter만으로 화면과 그래프를 제공하며 결과는 공유용 텍스트로 저장한다.
"""

from __future__ import annotations

import json
import math
import threading
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from tkinter import (
    BOTH, END, LEFT, RIGHT, VERTICAL, BooleanVar, Button, Canvas, Checkbutton,
    Entry, Frame, Label, Listbox, Radiobutton, Scrollbar, StringVar, Tk, Toplevel,
    filedialog, messagebox, ttk,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
BUILD = "lab-1.0-20260814"

BG = "#0b0f14"
PANEL = "#121a23"
PANEL2 = "#182331"
LINE = "#2a394a"
TEXT = "#edf4fb"
MUTED = "#96a7ba"
BLUE = "#4aa3ff"
GREEN = "#51d88a"
YELLOW = "#f1c75b"
RED = "#ff6f7d"

KOREAN_GROUPS = {
    "momentum": "모멘텀", "trend": "추세", "risk": "위험·저변동성",
    "volume": "거래량·거래대금", "oscillator": "보조지표",
    "price_action": "가격행동", "flow": "투자자 수급", "dart": "DART 재무",
}

NAME_HINTS = {
    "ret": "기간 수익률", "momentum": "모멘텀", "ma": "이동평균",
    "vol": "변동성", "drawdown": "낙폭", "skew": "왜도", "value": "거래대금",
    "volume": "거래량", "rsi": "RSI", "macd": "MACD", "gap": "갭 위험",
    "foreign": "외국인", "institution": "기관", "individual": "개인",
    "cashflow": "현금흐름", "margin": "이익률", "growth": "성장률",
    "debt": "부채", "roe": "자기자본수익성", "roic": "자산수익성",
}


@dataclass
class Rule:
    side: str
    indicator: str
    operator: str
    value: float
    parameter: str = ""


def korean_name(name: str) -> str:
    text = name
    for source, target in NAME_HINTS.items():
        text = text.replace(source, target)
    return text.replace("_", " ")


def month_end_positions(index):
    import pandas as pd
    series = pd.Series(range(len(index)), index=index)
    return series.groupby(index.to_period("M")).last().to_numpy(dtype=int)


def custom_rsi(close, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


class ResearchEngine:
    def __init__(self, progress):
        self.progress = progress
        self.loaded = False
        self.catalog: list[dict[str, str]] = []

    def load(self):
        if self.loaded:
            return
        import pandas as pd
        import seedforge_021 as core
        import seedforge_031 as search
        from seedforge_029 import find_flow_file, load_flow

        self.progress("가격·수급·재무 데이터를 불러오는 중...")
        self.market, self.features, self.overbought, self.neutral, self.kospi = core.load_or_prepare(False)
        self.close = pd.DataFrame(self.market.close, index=self.market.dates, columns=self.market.tickers)
        self.open = pd.DataFrame(self.market.open, index=self.market.dates, columns=self.market.tickers)
        self.volume = pd.DataFrame(self.market.volume, index=self.market.dates, columns=self.market.tickers)
        flows = load_flow(find_flow_file(), self.close.columns)
        self.dates = pd.DatetimeIndex(flows["외국인"].index)
        self.progress("기술·수급·DART 지표 등록소를 만드는 중...")
        arrays, groups = search.build_factor_library(self.close, self.open, self.volume, self.dates, flows)
        self.arrays = arrays
        self.groups = groups
        self.catalog = [
            {"id": name, "name": korean_name(name), "group": KOREAN_GROUPS.get(groups[name], groups[name]),
             "description": "전체 종목 내 백분위 순위. 0은 낮음, 1은 높음."}
            for name in sorted(arrays)
        ]
        self.catalog.extend([
            {"id": "CUSTOM_RSI", "name": "사용자 RSI", "group": "보조지표",
             "description": "기간을 직접 정하는 RSI 원값(0~100)."},
            {"id": "RSI_NEUTRAL_COUNT", "name": "중립 약세 다이버전스 횟수", "group": "RSI 상세",
             "description": "최근 지정 거래일 동안 RSI 30~70 약세 다이버전스 발생 횟수."},
            {"id": "RSI_OVERBOUGHT_COUNT", "name": "과매수 약세 다이버전스 횟수", "group": "RSI 상세",
             "description": "최근 지정 거래일 동안 RSI 70 이상 약세 다이버전스 발생 횟수. 2 이상은 더블, 3 이상은 트리플."},
            {"id": "RSI_ALL_BEAR_COUNT", "name": "전체 약세 다이버전스 횟수", "group": "RSI 상세",
             "description": "중립과 과매수 약세 다이버전스를 합친 횟수."},
        ])
        self.loaded = True
        self.progress(f"준비 완료: 지표 {len(self.catalog)}개")

    def indicator_matrix(self, rule: Rule):
        import numpy as np
        if rule.indicator in self.arrays:
            return self.arrays[rule.indicator]
        if rule.indicator == "CUSTOM_RSI":
            period = max(int(float(rule.parameter or 14)), 2)
            frame = custom_rsi(self.close, period)
            positions = self.close.index.searchsorted(self.dates, side="right") - 1
            return frame.iloc[positions].to_numpy(dtype="float32")
        lookback = max(int(float(rule.parameter or 60)), 1)
        neutral = rule.indicator in {"RSI_NEUTRAL_COUNT", "RSI_ALL_BEAR_COUNT"}
        overbought = rule.indicator in {"RSI_OVERBOUGHT_COUNT", "RSI_ALL_BEAR_COUNT"}
        result = np.zeros((len(self.dates), len(self.market.tickers)), dtype="float32")
        signal_days = []
        for date in self.dates:
            signal_days.append(int(self.market.dates.searchsorted(date, side="right") - 1))
        for row, day in enumerate(signal_days):
            start = max(0, day - lookback + 1)
            for source in range(start, day + 1):
                if neutral:
                    for stock in self.neutral.get(source, ()):
                        result[row, stock] += 1
                if overbought:
                    for stock in self.overbought.get(source, ()):
                        result[row, stock] += 1
        return result

    @staticmethod
    def apply_rule(values, rule: Rule):
        import numpy as np
        finite = np.isfinite(values)
        if rule.operator == "상위 %":
            threshold = np.nanquantile(values, 1 - rule.value / 100, axis=1)[:, None]
            return finite & (values >= threshold)
        if rule.operator == "하위 %":
            threshold = np.nanquantile(values, rule.value / 100, axis=1)[:, None]
            return finite & (values <= threshold)
        if rule.operator == "이상":
            return finite & (values >= rule.value)
        if rule.operator == "이하":
            return finite & (values <= rule.value)
        if rule.operator == "범위 안":
            low, high = sorted((rule.value, float(rule.parameter or rule.value)))
            return finite & (values >= low) & (values <= high)
        raise ValueError(f"지원하지 않는 비교 방식: {rule.operator}")

    def combined_mask(self, rules: list[Rule], logic: str):
        import numpy as np
        masks = [self.apply_rule(self.indicator_matrix(rule), rule) for rule in rules]
        if not masks:
            return np.zeros((len(self.dates), len(self.market.tickers)), dtype=bool)
        result = masks[0].copy()
        for mask in masks[1:]:
            result = result & mask if logic == "모두 충족(AND)" else result | mask
        return result

    def forward_returns(self, horizons_weeks: list[int]):
        import numpy as np
        entry = self.close.index.searchsorted(self.dates, side="right")
        results = {}
        for weeks in horizons_weeks:
            exit_pos = entry + weeks * 5 - 1
            valid_rows = exit_pos < len(self.close)
            matrix = np.full((len(self.dates), len(self.close.columns)), np.nan, dtype="float32")
            if valid_rows.any():
                entries = self.open.iloc[entry[valid_rows]].to_numpy(dtype=float)
                exits = self.close.iloc[exit_pos[valid_rows]].to_numpy(dtype=float)
                matrix[valid_rows] = exits / entries - 1
            results[weeks] = matrix
        return results

    def run(self, mode: str, rules: list[Rule], logic: str, horizons: list[int]):
        import numpy as np
        self.load()
        buy_rules = [r for r in rules if r.side == "매수"]
        sell_rules = [r for r in rules if r.side == "매도"]
        buy_mask = self.combined_mask(buy_rules, logic)
        sell_mask = self.combined_mask(sell_rules, logic)
        targets = self.forward_returns(horizons)
        rows = []
        date_masks = {
            "발견(2014~2017)": np.asarray(self.dates <= "2017-12-31"),
            "검증(2018~2021)": np.asarray((self.dates >= "2018-01-01") & (self.dates <= "2021-12-31")),
            "열린 진단(2022~)": np.asarray(self.dates >= "2022-01-01"),
        }
        for weeks in horizons:
            target = targets[weeks]
            if mode == "매수 테스트":
                base_mask, values = buy_mask & np.isfinite(target), target
            elif mode == "매도 테스트":
                base_mask, values = sell_mask & np.isfinite(target), -target
            else:
                # 결합 모드는 매수 대상 중 같은 시점 매도조건이 없는 종목을 보유한다.
                base_mask, values = buy_mask & ~sell_mask & np.isfinite(target), target
            for sample_name, date_mask in date_masks.items():
                samples = values[base_mask & date_mask[:, None]]
                rows.append({
                    "sample": sample_name, "weeks": weeks, "count": int(samples.size),
                    "mean": float(np.nanmean(samples)) if samples.size else math.nan,
                    "median": float(np.nanmedian(samples)) if samples.size else math.nan,
                    "positive": float(np.mean(samples > 0)) if samples.size else math.nan,
                    "p25": float(np.nanquantile(samples, .25)) if samples.size else math.nan,
                    "p75": float(np.nanquantile(samples, .75)) if samples.size else math.nan,
                })
        return {"build": BUILD, "mode": mode, "logic": logic, "rules": [asdict(r) for r in rules],
                "generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows,
                "note": "열린 구간을 분리한 최종 실전 판정이 아니라 전체 사건연구 진단입니다."}


class RuleDialog(Toplevel):
    def __init__(self, master, catalog, callback):
        super().__init__(master)
        self.title("조건 추가")
        self.configure(bg=BG)
        self.geometry("640x420")
        self.callback = callback
        self.catalog = catalog
        self.side = StringVar(value="매수")
        self.group = StringVar(value="전체")
        self.indicator = StringVar()
        self.operator = StringVar(value="상위 %")
        self.value = StringVar(value="20")
        self.parameter = StringVar(value="14")
        self._build()

    def _build(self):
        body = Frame(self, bg=PANEL, padx=20, pady=20)
        body.pack(fill=BOTH, expand=True, padx=14, pady=14)
        groups = ["전체"] + sorted({item["group"] for item in self.catalog})
        fields = [
            ("용도", ttk.Combobox(body, textvariable=self.side, values=["매수", "매도"], state="readonly")),
            ("지표 계열", ttk.Combobox(body, textvariable=self.group, values=groups, state="readonly")),
            ("세부 지표", ttk.Combobox(body, textvariable=self.indicator, state="readonly", width=48)),
            ("비교 방식", ttk.Combobox(body, textvariable=self.operator,
                values=["상위 %", "하위 %", "이상", "이하", "범위 안"], state="readonly")),
            ("기준값", Entry(body, textvariable=self.value, bg=BG, fg=TEXT, insertbackground=TEXT)),
            ("세부값(기간 또는 범위 상한)", Entry(body, textvariable=self.parameter, bg=BG, fg=TEXT, insertbackground=TEXT)),
        ]
        for row, (name, widget) in enumerate(fields):
            Label(body, text=name, bg=PANEL, fg=MUTED).grid(row=row, column=0, sticky="w", pady=8)
            widget.grid(row=row, column=1, sticky="ew", padx=12, pady=8)
        body.columnconfigure(1, weight=1)
        self.group.trace_add("write", lambda *_: self.refresh())
        Button(body, text="조건 추가", command=self.submit, bg=BLUE, fg="white", relief="flat", padx=18, pady=8).grid(row=7, column=1, sticky="e", pady=18)
        self.refresh()

    def refresh(self):
        items = [x for x in self.catalog if self.group.get() == "전체" or x["group"] == self.group.get()]
        names = [f'{x["name"]}  [{x["id"]}]' for x in items]
        combo = self.children["!frame"].grid_slaves(row=2, column=1)[0]
        combo["values"] = names
        if names:
            self.indicator.set(names[0])

    def submit(self):
        try:
            indicator = self.indicator.get().rsplit("[", 1)[1].rstrip("]")
            rule = Rule(self.side.get(), indicator, self.operator.get(), float(self.value.get()), self.parameter.get())
        except (ValueError, IndexError):
            messagebox.showerror("입력 오류", "지표와 기준값을 확인하세요.", parent=self)
            return
        self.callback(rule)
        self.destroy()


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("SeedForge 무코드 전략 실험실")
        self.root.geometry("1380x860")
        self.root.configure(bg=BG)
        self.mode = StringVar(value="매수 테스트")
        self.logic = StringVar(value="모두 충족(AND)")
        self.status = StringVar(value="지표 준비 전")
        self.rules: list[Rule] = []
        self.result = None
        self.engine = ResearchEngine(self.set_status)
        self._style()
        self._build()
        threading.Thread(target=self.prepare, daemon=True).start()

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=BG, background=PANEL2, foreground=TEXT)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=29)
        style.configure("Treeview.Heading", background=PANEL2, foreground=TEXT)

    def _build(self):
        header = Frame(self.root, bg=BG, height=70)
        header.pack(fill="x", padx=26, pady=(16, 4))
        Label(header, text="SeedForge 무코드 전략 실험실", bg=BG, fg=TEXT, font=("Malgun Gothic", 20, "bold")).pack(side=LEFT)
        Label(header, text="매수 · 매도 · 동시 테스트", bg=BG, fg=BLUE, font=("Malgun Gothic", 11, "bold")).pack(side=LEFT, padx=18)
        Label(header, textvariable=self.status, bg=PANEL2, fg=GREEN, padx=12, pady=7).pack(side=RIGHT)

        modebar = Frame(self.root, bg=PANEL, padx=18, pady=12)
        modebar.pack(fill="x", padx=26, pady=8)
        for mode in ["매수 테스트", "매도 테스트", "매수/매도 동시 테스트"]:
            Radiobutton(modebar, text=mode, variable=self.mode, value=mode, bg=PANEL, fg=TEXT,
                        selectcolor=PANEL2, activebackground=PANEL, activeforeground=TEXT).pack(side=LEFT, padx=10)
        ttk.Combobox(modebar, textvariable=self.logic, values=["모두 충족(AND)", "하나 이상 충족(OR)"], state="readonly", width=20).pack(side=RIGHT)
        Label(modebar, text="조건 결합", bg=PANEL, fg=MUTED).pack(side=RIGHT, padx=8)

        main = Frame(self.root, bg=BG)
        main.pack(fill=BOTH, expand=True, padx=26, pady=8)
        left = Frame(main, bg=PANEL, padx=16, pady=16)
        left.pack(side=LEFT, fill=BOTH, expand=False)
        right = Frame(main, bg=PANEL, padx=16, pady=16)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(14, 0))

        Label(left, text="조건식", bg=PANEL, fg=TEXT, font=("Malgun Gothic", 14, "bold")).pack(anchor="w")
        Label(left, text="지표와 세부값을 자유롭게 추가합니다.", bg=PANEL, fg=MUTED).pack(anchor="w", pady=(0, 10))
        self.rule_list = Listbox(left, width=58, height=22, bg=BG, fg=TEXT, selectbackground="#244d72", relief="flat")
        self.rule_list.pack(fill=BOTH, expand=True)
        actions = Frame(left, bg=PANEL)
        actions.pack(fill="x", pady=10)
        Button(actions, text="+ 조건 추가", command=self.add_rule, bg=BLUE, fg="white", relief="flat", padx=12, pady=7).pack(side=LEFT)
        Button(actions, text="선택 삭제", command=self.remove_rule, bg=PANEL2, fg=TEXT, relief="flat", padx=12, pady=7).pack(side=LEFT, padx=7)
        Button(actions, text="전체 삭제", command=self.clear_rules, bg="#3b2027", fg=RED, relief="flat", padx=12, pady=7).pack(side=LEFT)

        horizon = Frame(left, bg=PANEL)
        horizon.pack(fill="x", pady=8)
        Label(horizon, text="확인 기간(주)", bg=PANEL, fg=MUTED).pack(anchor="w")
        self.horizon_vars = {}
        for value in [1, 2, 4, 8, 12, 26, 52]:
            var = BooleanVar(value=True)
            self.horizon_vars[value] = var
            Checkbutton(horizon, text=str(value), variable=var, bg=PANEL, fg=TEXT, selectcolor=PANEL2,
                        activebackground=PANEL, activeforeground=TEXT).pack(side=LEFT)

        Button(left, text="테스트 시작", command=self.start_test, bg=GREEN, fg="#06140b", relief="flat",
               font=("Malgun Gothic", 11, "bold"), pady=10).pack(fill="x", pady=(12, 5))
        Button(left, text="결과를 피드백용 텍스트로 저장", command=self.save_text, bg=YELLOW, fg="#1b1606", relief="flat", pady=9).pack(fill="x")

        Label(right, text="기간별 결과 그래프", bg=PANEL, fg=TEXT, font=("Malgun Gothic", 14, "bold")).pack(anchor="w")
        Label(right, text="매수는 이후 수익률, 매도는 이후 하락 회피수익을 표시합니다.", bg=PANEL, fg=MUTED).pack(anchor="w")
        self.canvas = Canvas(right, bg=BG, highlightthickness=1, highlightbackground=LINE, height=360)
        self.canvas.pack(fill="x", pady=12)
        columns = ("구간", "기간", "표본", "평균", "중앙", "양수비율", "하위25%", "상위25%")
        self.table = ttk.Treeview(right, columns=columns, show="headings", height=11)
        for column in columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=125 if column == "구간" else (70 if column == "기간" else 88), anchor="center")
        scroll = Scrollbar(right, orient=VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side=LEFT, fill=BOTH, expand=True)
        scroll.pack(side=RIGHT, fill="y")

    def set_status(self, text):
        self.root.after(0, self.status.set, text)

    def prepare(self):
        try:
            self.engine.load()
        except Exception:
            self.set_status("준비 실패")
            self.root.after(0, messagebox.showerror, "지표 준비 오류", traceback.format_exc())

    def add_rule(self):
        if not self.engine.loaded:
            messagebox.showinfo("준비 중", "지표를 준비하고 있습니다. 잠시 후 다시 눌러주세요.")
            return
        RuleDialog(self.root, self.engine.catalog, self.append_rule)

    def append_rule(self, rule):
        self.rules.append(rule)
        self.rule_list.insert(END, f"[{rule.side}] {korean_name(rule.indicator)} · {rule.operator} {rule.value:g} · 세부 {rule.parameter or '-'}")

    def remove_rule(self):
        selected = list(self.rule_list.curselection())
        for index in reversed(selected):
            self.rule_list.delete(index)
            del self.rules[index]

    def clear_rules(self):
        self.rules.clear()
        self.rule_list.delete(0, END)

    def start_test(self):
        if not self.rules:
            messagebox.showwarning("조건 없음", "매수 또는 매도 조건을 하나 이상 추가하세요.")
            return
        mode = self.mode.get()
        if mode == "매수 테스트" and not any(r.side == "매수" for r in self.rules):
            messagebox.showwarning("매수 조건 없음", "매수 조건을 추가하세요.")
            return
        if mode == "매도 테스트" and not any(r.side == "매도" for r in self.rules):
            messagebox.showwarning("매도 조건 없음", "매도 조건을 추가하세요.")
            return
        horizons = [n for n, var in self.horizon_vars.items() if var.get()]
        self.set_status("테스트 실행 중...")
        threading.Thread(target=self._run, args=(mode, horizons), daemon=True).start()

    def _run(self, mode, horizons):
        try:
            result = self.engine.run(mode, self.rules, self.logic.get(), horizons)
            self.result = result
            self.root.after(0, self.show_result)
            self.set_status("테스트 완료")
        except Exception:
            self.set_status("테스트 실패")
            self.root.after(0, messagebox.showerror, "테스트 오류", traceback.format_exc())

    def show_result(self):
        for item in self.table.get_children():
            self.table.delete(item)
        for row in self.result["rows"]:
            pct = lambda x: "-" if not math.isfinite(x) else f"{x:.2%}"
            self.table.insert("", END, values=(row["sample"], f'{row["weeks"]}주', f'{row["count"]:,}', pct(row["mean"]),
                pct(row["median"]), pct(row["positive"]), pct(row["p25"]), pct(row["p75"])))
        self.draw_chart()

    def draw_chart(self):
        self.canvas.delete("all")
        rows = [r for r in self.result["rows"] if r["sample"].startswith("검증") and math.isfinite(r["mean"])]
        if not rows:
            self.canvas.create_text(400, 170, text="표시할 결과가 없습니다.", fill=MUTED)
            return
        self.canvas.update_idletasks()
        width, height = max(self.canvas.winfo_width(), 600), 360
        left, right, top, bottom = 65, width - 25, 25, height - 45
        values = [r["mean"] for r in rows] + [r["median"] for r in rows] + [0]
        low, high = min(values), max(values)
        margin = max((high - low) * .15, .01)
        low, high = low - margin, high + margin
        def xy(i, value):
            x = left + (right - left) * i / max(len(rows) - 1, 1)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            return x, y
        zero_y = xy(0, 0)[1]
        self.canvas.create_line(left, zero_y, right, zero_y, fill="#526274", dash=(4, 4))
        for color, key, name in [(BLUE, "mean", "평균"), (GREEN, "median", "중앙값")]:
            points = [xy(i, row[key]) for i, row in enumerate(rows)]
            self.canvas.create_line(*sum(([x, y] for x, y in points), []), fill=color, width=3, smooth=True)
            for x, y in points:
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, outline="")
            self.canvas.create_text(right - 90, top + (18 if key == "mean" else 40), text=name, fill=color, anchor="w")
        for i, row in enumerate(rows):
            x, _ = xy(i, 0)
            self.canvas.create_text(x, bottom + 20, text=f'{row["weeks"]}주', fill=MUTED)
        self.canvas.create_text(12, top, text=f"{high:.1%}", fill=MUTED, anchor="w")
        self.canvas.create_text(12, bottom, text=f"{low:.1%}", fill=MUTED, anchor="w")

    def save_text(self):
        if not self.result:
            messagebox.showinfo("결과 없음", "먼저 테스트를 실행하세요.")
            return
        default = f'seedforge_피드백_{datetime.now():%Y%m%d_%H%M}.txt'
        path = filedialog.asksaveasfilename(initialdir=RESULTS, initialfile=default,
            defaultextension=".txt", filetypes=[("텍스트 파일", "*.txt")])
        if not path:
            return
        lines = ["SeedForge 피드백용 결과", f'생성: {self.result["generated_at"]}',
                 f'모드: {self.result["mode"]}', f'조건 결합: {self.result["logic"]}', "", "조건:"]
        for rule in self.rules:
            lines.append(f'- [{rule.side}] {korean_name(rule.indicator)} / {rule.operator} {rule.value:g} / 세부값 {rule.parameter or "없음"}')
        lines.extend(["", "기간별 결과:", "구간\t기간\t표본\t평균\t중앙값\t양수비율\t하위25%\t상위25%"])
        for row in self.result["rows"]:
            lines.append(f'{row["sample"]}\t{row["weeks"]}주\t{row["count"]}\t{row["mean"]:.6f}\t{row["median"]:.6f}\t{row["positive"]:.6f}\t{row["p25"]:.6f}\t{row["p75"]:.6f}')
        lines.extend(["", "주의:", self.result["note"], "이 파일을 AI 대화에 첨부해 결과 피드백을 요청하세요."])
        Path(path).write_text("\n".join(lines), encoding="utf-8-sig")
        messagebox.showinfo("저장 완료", f"피드백용 파일을 저장했습니다.\n{path}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
