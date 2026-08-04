print("SeedForge 006 파일 실행 시작", flush=True)
"""RSI 3저점 다이버전스 백테스트 (거래 원장/시간축 수정판).

핵심 변경점
* 지표는 적격성 마스크를 씌우지 않은 원시 가격의 전체 거래일 축에서 계산한다.
* 적격 조건은 신규 진입에만 적용하며, 유동성 정보는 하루 지연한다.
* 종가로 확인한 신호는 다음 거래일 시가에 체결한다.
* 현금 기반 포트폴리오와 거래 원장을 함께 만들어 거래별 통계를 계산한다.

데이터에 수정주가가 들어 있다는 전제다. 상장폐지 수익률이 별도 제공되는 데이터라면
``load_data``에서 마지막 가격에 그 수익률을 반영해야 한다.
"""

print("SeedForge 006 모듈 로딩 중...", flush=True)

import importlib.util
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable


def ensure_dependencies() -> None:
    missing = [name for name in ("numpy", "pandas", "pyarrow") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    packages = " ".join(missing)
    print("\n필수 패키지가 현재 Python 환경에 없습니다:", ", ".join(missing), flush=True)
    print("아래 명령을 같은 창에서 실행한 뒤 다시 시도하세요:", flush=True)
    print(f"  {sys.executable} -m pip install {packages}", flush=True)
    print("가상환경을 쓰는 경우 먼저 venv\\Scripts\\activate를 실행하세요.", flush=True)
    raise SystemExit(1)


ensure_dependencies()

import numpy as np
import pandas as pd


D = Path("./data")
CACHE = D / "cache" / "seedforge_006_prepared.pkl"
PERIODS = (
    ("2014_2017", "2014-05-12", "2017-12-31"),
    ("2018_2020", "2018-01-01", "2020-12-31"),
    ("2021_2023", "2021-01-01", "2023-12-31"),
    ("2024_2026", "2024-01-01", "2026-12-31"),
    ("train_2014_2021", "2014-05-12", "2021-12-31"),
    ("test_2022_2026", "2022-01-01", "2026-12-31"),
)
SLOTS, COST = 20, 0.0058
COST_PER_SIDE = COST / 2  # 기존 COST는 왕복비용으로 해석한다.
OS, OB, K, FLOOR, GAP = 30, 70, 5, "2014-05-12", 60
INITIAL_CAPITAL = 1.0


def rsi(price: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. NaN 날짜를 없애지 않아 거래일 간격을 보존한다."""
    delta = price.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def piv(values: np.ndarray, k: int, low: bool = True) -> list[tuple[int, int]]:
    """피봇 날짜와 k일 뒤 확인 날짜를 반환한다. 결측치를 건너뛰지 않는다."""
    out: list[tuple[int, int]] = []
    for i in range(k, len(values) - k):
        window = values[i - k : i + k + 1]
        if np.isnan(window).any():
            continue
        target = np.min(window) if low else np.max(window)
        if values[i] == target:
            out.append((i, i + k))
    return out


def rsi_frame(price: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """모든 종목의 Wilder RSI를 DataFrame 단위로 한 번에 계산한다."""
    delta = price.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def pivot_mask(frame: pd.DataFrame, k: int, low: bool = True) -> pd.DataFrame:
    """피봇 여부를 전체 종목에 대해 벡터화해서 계산한다."""
    window = k * 2 + 1
    roller = frame.rolling(window=window, center=True, min_periods=window)
    target = roller.min() if low else roller.max()
    return frame.eq(target) & target.notna()


def generate_signals(
    price: np.ndarray,
    strength: np.ndarray,
    volume: np.ndarray,
    relative_strength: np.ndarray,
    lows: list[tuple[int, int]],
    highs: list[tuple[int, int]],
    reentry_rsi: int,
    confirm_days: int,
    volume_factor: float,
    use_relative_strength: bool,
) -> tuple[list[tuple[int, float]], set[int], set[int]]:
    """모든 반환 날짜는 신호를 종가로 확인할 수 있는 날짜다."""
    n = len(price)
    buys: list[tuple[int, float]] = []
    for a in range(len(lows) - 2):
        i1, i2, i3 = lows[a][0], lows[a + 1][0], lows[a + 2][0]
        confirmed = lows[a + 2][1]
        if confirmed >= n or i3 - i1 > GAP * 2:
            continue
        if not (price[i1] > price[i2] > price[i3]):
            continue
        if not (strength[i1] < strength[i2] < strength[i3]):
            continue
        if not (
            strength[i1] < OS
            and strength[i2] < OS
            and OS <= strength[i3] < OB
        ):
            continue
        for j in range(confirmed + 1, min(n, confirmed + confirm_days + 1)):
            if not np.isfinite(price[j]) or strength[j] < reentry_rsi:
                continue
            prior = price[confirmed:j]
            if np.isnan(prior).any() or not price[j] > np.max(prior):
                continue
            if use_relative_strength and not (
                np.isfinite(relative_strength[j]) and relative_strength[j] > 0
            ):
                continue
            if volume_factor > 1.0:
                if j < 20 or np.isnan(volume[j - 20 : j + 1]).any():
                    continue
                average = np.mean(volume[j - 20 : j])
                if not (average > 0 and volume[j] > average * volume_factor):
                    continue
            buys.append((j, strength[j] - strength[confirmed]))
            break

    overbought: set[int] = set()
    neutral: set[int] = set()
    for a in range(len(highs) - 1):
        j1, j2, confirmed = highs[a][0], highs[a + 1][0], highs[a + 1][1]
        if confirmed >= n or j2 - j1 > GAP:
            continue
        if not (price[j1] < price[j2] and strength[j1] > strength[j2]):
            continue
        if strength[j1] > OB and strength[j2] > OB:
            overbought.add(confirmed)
        elif OS <= strength[j2] < OB:
            neutral.add(confirmed)
    return buys, overbought, neutral


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_cost: float
    initial_shares: float
    breakeven_armed: bool = False
    overbought_exits: int = 0
    gross_proceeds: float = 0.0
    exit_cost: float = 0.0
    partial_exits: list[dict[str, object]] = field(default_factory=list)
    mfe: float = 0.0
    mae: float = 0.0
    pending_exit: str | None = None


@dataclass(frozen=True)
class StockFeatures:
    """모든 조합에서 재사용하는 종목별 지표와 피봇."""

    column: int
    price: np.ndarray
    strength: np.ndarray
    volume: np.ndarray
    relative_strength: np.ndarray
    lows: list[tuple[int, int]]
    highs: list[tuple[int, int]]


@dataclass(frozen=True)
class SimulationInputs:
    """36회 시뮬레이션에서 반복 변환하지 않을 시장 배열."""

    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    close: np.ndarray
    open: np.ndarray
    marked_close: np.ndarray
    entry_eligible: np.ndarray


def _sell(
    position: Position,
    fraction: float,
    date: pd.Timestamp,
    price: float,
    reason: str,
) -> tuple[float, bool]:
    shares = position.shares if fraction >= 1 else position.shares * fraction
    gross = shares * price
    fee = gross * COST_PER_SIDE
    position.shares -= shares
    position.gross_proceeds += gross
    position.exit_cost += fee
    position.partial_exits.append(
        {"date": date, "price": price, "shares": shares, "reason": reason, "cost": fee}
    )
    return gross - fee, position.shares <= position.initial_shares * 1e-10


def _close_trade(position: Position) -> dict[str, object]:
    last = position.partial_exits[-1]
    invested = position.initial_shares * position.entry_price
    pnl = position.gross_proceeds - invested - position.entry_cost - position.exit_cost
    return {
        "ticker": position.ticker,
        "entry_date": position.entry_date,
        "entry_price": position.entry_price,
        "exit_date": last["date"],
        "exit_price": last["price"],
        "exit_reason": last["reason"],
        "shares": position.initial_shares,
        "entry_cost": position.entry_cost,
        "exit_cost": position.exit_cost,
        "gross_pnl": position.gross_proceeds - invested,
        "net_pnl": pnl,
        "return": pnl / (invested + position.entry_cost),
        "holding_days": (last["date"] - position.entry_date).days,
        "mae": position.mae,
        "mfe": position.mfe,
        "partial_exits": position.partial_exits,
    }


def simulate(
    market: SimulationInputs,
    buy_signals: dict[int, list[tuple[int, float]]],
    overbought_signals: dict[int, set[int]],
    neutral_signals: dict[int, set[int]],
    stop_loss: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, int]]:
    """신호 다음 날 시가 체결. 원장 순손익과 현금 변화가 같은 회계에 속한다."""
    dates, tickers = market.dates, market.tickers
    close_values, open_values = market.close, market.open
    marked_close_values = market.marked_close
    eligible_values = market.entry_eligible
    cash, positions = INITIAL_CAPITAL, {}
    trades: list[dict[str, object]] = []
    equity, exposure = [], []
    signal_count = filled_count = skipped_count = 0

    # 종가 확인 신호를 다음 거래일 실행 이벤트로 옮긴다.
    buys_at: dict[int, list[tuple[int, float]]] = {}
    overbought_at: dict[int, set[int]] = {}
    neutral_at: dict[int, set[int]] = {}
    for day, signals in buy_signals.items():
        if day + 1 < len(dates):
            buys_at.setdefault(day + 1, []).extend(signals)
    for source, target in (
        (overbought_signals, overbought_at),
        (neutral_signals, neutral_at),
    ):
        for day, names in source.items():
            if day + 1 < len(dates):
                target.setdefault(day + 1, set()).update(names)

    for i, date in enumerate(dates):
        # 매도 신호는 매수보다 먼저 다음 날 시가에 실행한다.
        for j in list(positions):
            position = positions[j]
            opening = open_values[i, j]
            previous_close = close_values[i - 1, j] if i > 0 else np.nan
            if j in neutral_at.get(i, set()):
                position.pending_exit = "neutral_divergence"
            elif (
                position.breakeven_armed
                and np.isfinite(previous_close)
                and previous_close <= position.entry_price
            ):
                position.pending_exit = "breakeven_stop"
            elif (
                stop_loss is not None
                and np.isfinite(previous_close)
                and previous_close <= position.entry_price * (1 + stop_loss)
            ):
                position.pending_exit = "price_stop"
            elif j in overbought_at.get(i, set()):
                position.pending_exit = "overbought_half"
            if not (np.isfinite(opening) and opening > 0) or position.pending_exit is None:
                continue  # 거래정지: 예약한 청산을 체결 가능일까지 유지한다.
            reason = position.pending_exit
            fraction = 0.5 if reason == "overbought_half" else 1.0
            if reason == "overbought_half":
                planned_shares = position.shares * fraction
                remaining = position.shares - planned_shares
                if remaining < position.initial_shares * 0.20:
                    fraction = 1.0
            proceeds, closed = _sell(position, fraction, date, opening, reason)
            position.pending_exit = None
            if reason == "overbought_half":
                position.breakeven_armed = True
                position.overbought_exits += 1
            cash += proceeds
            if closed:
                trades.append(_close_trade(position))
                del positions[j]

        candidates = buys_at.get(i, [])
        signal_count += len(candidates)
        for j, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
            opening = open_values[i, j]
            if (
                j in positions
                or not eligible_values[i, j]
                or not (np.isfinite(opening) and opening > 0)
            ):
                skipped_count += 1
                continue
            if len(positions) >= SLOTS:
                skipped_count += 1
                continue
            budget = min(INITIAL_CAPITAL / SLOTS, cash / (1 + COST_PER_SIDE))
            if budget <= 0:
                skipped_count += 1
                continue
            shares = budget / opening
            fee = budget * COST_PER_SIDE
            cash -= budget + fee
            positions[j] = Position(
                ticker=tickers[j],
                entry_date=date,
                entry_price=opening,
                shares=shares,
                initial_shares=shares,
                entry_cost=fee,
            )
            filled_count += 1

        market_value = 0.0
        for j, position in positions.items():
            price = close_values[i, j]
            if not np.isfinite(price):
                # 정지/결측일은 마지막 관측 가격으로 보수적으로 동결한다.
                price = marked_close_values[i, j]
                if not np.isfinite(price):
                    price = position.entry_price
            move = price / position.entry_price - 1
            position.mfe = max(position.mfe, move)
            position.mae = min(position.mae, move)
            market_value += position.shares * price
        total = cash + market_value
        equity.append(total)
        exposure.append(market_value / total if total > 0 else np.nan)

    # 마지막 종가에 강제청산하여 열린 거래와 최종 비용을 빠뜨리지 않는다.
    final_date = dates[-1]
    for j in list(positions):
        position = positions[j]
        price = marked_close_values[-1, j]
        if not np.isfinite(price):
            price = position.entry_price
        proceeds, _ = _sell(position, 1.0, final_date, price, "end_of_test")
        cash += proceeds
        trades.append(_close_trade(position))
        del positions[j]
    equity[-1] = cash
    exposure[-1] = 0.0

    stats = {"signals": signal_count, "filled": filled_count, "skipped": skipped_count}
    return (
        pd.Series(equity, index=dates, name="equity"),
        pd.Series(exposure, index=dates, name="exposure"),
        pd.DataFrame(trades),
        stats,
    )


def performance(
    equity: pd.Series,
    exposure: pd.Series,
    benchmark_return: pd.Series,
    trades: pd.DataFrame,
) -> dict[str, float]:
    returns = equity.pct_change(fill_method=None).fillna(0)
    benchmark = benchmark_return.reindex(returns.index).fillna(0)
    matched_benchmark = benchmark * exposure.shift(1).fillna(0)
    excess = returns - matched_benchmark
    years = max(len(returns) / 252, 1 / 252)
    curve = equity / equity.iloc[0]
    result = {
        "cagr": curve.iloc[-1] ** (1 / years) - 1,
        "mdd": (curve / curve.cummax() - 1).min(),
        "exposure": exposure.mean(),
        "t_exposure_matched": (
            excess.mean() / excess.std(ddof=1) * np.sqrt(len(excess))
            if excess.std(ddof=1) > 0
            else np.nan
        ),
    }
    if trades.empty:
        result.update(win_rate=np.nan, payoff=np.nan, trade_pf=np.nan, expectancy=np.nan)
        return result
    winners = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losers = trades.loc[trades["net_pnl"] < 0, "net_pnl"]
    result.update(
        win_rate=len(winners) / len(trades),
        payoff=(winners.mean() / abs(losers.mean()) if len(winners) and len(losers) else np.nan),
        trade_pf=(winners.sum() / abs(losers.sum()) if len(losers) else np.nan),
        expectancy=trades["net_pnl"].mean(),
    )
    return result


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    files = sorted((D / "ohlcv").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("data/ohlcv에 parquet 파일이 없습니다")
    print(f"OHLCV parquet {len(files):,}개 읽는 중...", flush=True)
    frames = []
    for number, path in enumerate(files, 1):
        frames.append(pd.read_parquet(path))
        if number == 1 or number % 10 == 0 or number == len(files):
            print(f"  - {number:,}/{len(files):,} 파일 로드 완료: {path.name}", flush=True)
    print("OHLCV 병합 중...", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] >= FLOOR]
    for column in ("시가", "고가", "저가", "종가"):
        if column in frame:
            frame.loc[frame[column] <= 0, column] = np.nan
    if "거래대금" not in frame:
        frame["거래대금"] = frame["종가"] * frame.get("거래량", 0)

    def pivot(column: str) -> pd.DataFrame:
        return frame.pivot_table(index="date", columns="ticker", values=column).sort_index()

    close = pivot("종가")
    if "시가" not in frame:
        raise ValueError("다음 거래일 체결을 검증하려면 OHLCV 데이터에 '시가'가 필요합니다")
    open_ = pivot("시가").reindex_like(close)
    volume = pivot("거래량").reindex_like(close) if "거래량" in frame else pivot("거래대금").reindex_like(close)
    value = pivot("거래대금").reindex_like(close)
    if "tradable" in frame:
        tradable_raw = frame.pivot_table(
            index="date", columns="ticker", values="tradable", aggfunc="max"
        ).reindex_like(close)
        # object dtype fillna의 silent downcast FutureWarning을 피한다.
        tradable = tradable_raw.eq(True) | tradable_raw.eq(1)  # noqa: E712
    else:
        tradable = open_.notna() & close.notna()

    common = pd.DataFrame(
        np.broadcast_to(
            np.array([str(column)[-1] == "0" for column in close.columns]), close.shape
        ),
        index=close.index,
        columns=close.columns,
    )
    # 전일 종가와 전일까지의 유동성만 신규 진입 판단에 사용한다.
    known_eligible = (
        (close.shift(1) >= 1000)
        & (value.shift(1).rolling(60, min_periods=20).mean() >= 1e8)
        & common
    )
    entry_eligible = known_eligible & tradable & open_.notna()

    kospi_file = D / "index" / "KOSPI.parquet"
    if not kospi_file.exists():
        raise FileNotFoundError("data/index/KOSPI.parquet가 없습니다")
    print("KOSPI parquet 읽는 중...", flush=True)
    kospi = pd.read_parquet(kospi_file)
    kospi["date"] = pd.to_datetime(kospi["date"])
    column = "종가" if "종가" in kospi else kospi.columns[-1]
    kospi_close = kospi.set_index("date")[column].reindex(close.index).ffill()
    return close, open_, volume, entry_eligible, kospi_close


def prepare_market(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    entry_eligible: pd.DataFrame,
) -> SimulationInputs:
    """대형 DataFrame→NumPy 변환과 ffill을 전체 실행에서 한 번만 수행한다."""
    return SimulationInputs(
        dates=pd.DatetimeIndex(close.index),
        tickers=tuple(map(str, close.columns)),
        close=close.to_numpy(),
        open=open_.to_numpy(),
        marked_close=close.ffill().to_numpy(),
        entry_eligible=entry_eligible.to_numpy(dtype=bool),
    )


def prepare_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    kospi_close: pd.Series,
) -> tuple[list[StockFeatures], dict[int, set[int]], dict[int, set[int]]]:
    """RSI·상대수익률·피봇·매도신호를 한 번 계산해 36개 조합이 공유한다."""
    kospi_120 = kospi_close / kospi_close.shift(120) - 1
    relative = (close / close.shift(120) - 1).sub(kospi_120, axis=0)
    print("RSI 전체 종목 벡터 계산 중...", flush=True)
    strength_frame = rsi_frame(close)
    print("가격 피봇 전체 종목 벡터 계산 중...", flush=True)
    low_mask = pivot_mask(close, K, True).to_numpy()
    high_mask = pivot_mask(close, K, False).to_numpy()
    close_values = close.to_numpy()
    volume_values = volume.to_numpy()
    relative_values = relative.to_numpy()
    strength_values = strength_frame.to_numpy()
    features: list[StockFeatures] = []
    overbought: dict[int, set[int]] = {}
    neutral: dict[int, set[int]] = {}

    for j in range(close.shape[1]):
        price = close_values[:, j]
        if np.isfinite(price).sum() < 250:
            continue
        strength = strength_values[:, j]
        lows = [(int(i), int(i + K)) for i in np.flatnonzero(low_mask[:, j]) if i + K < len(price)]
        highs = [(int(i), int(i + K)) for i in np.flatnonzero(high_mask[:, j]) if i + K < len(price)]
        stock = StockFeatures(
            column=j,
            price=price,
            strength=strength,
            volume=volume_values[:, j],
            relative_strength=relative_values[:, j],
            lows=lows,
            highs=highs,
        )
        features.append(stock)
        # 매도 신호는 진입 매개변수와 무관하므로 여기서 한 번만 계산한다.
        _, ob, nu = generate_signals(
            stock.price,
            stock.strength,
            stock.volume,
            stock.relative_strength,
            stock.lows,
            stock.highs,
            0,
            0,
            1.0,
            False,
        )
        _put(overbought, ob, j)
        _put(neutral, nu, j)
    return features, overbought, neutral


def _put(mapping: dict[int, set[int]], days: Iterable[int], ticker: int) -> None:
    for day in days:
        mapping.setdefault(day, set()).add(ticker)


def run_one(
    market: SimulationInputs,
    features: list[StockFeatures],
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    kospi_close: pd.Series,
    reentry_rsi: int,
    confirm_days: int,
    volume_factor: float,
    use_relative_strength: bool,
    stop_loss: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    buys: dict[int, list[tuple[int, float]]] = {}

    for stock in features:
        buy, _, _ = generate_signals(
            stock.price,
            stock.strength,
            stock.volume,
            stock.relative_strength,
            stock.lows,
            [],  # 매도신호는 prepare_features()에서 이미 계산했다.
            reentry_rsi,
            confirm_days,
            volume_factor,
            use_relative_strength,
        )
        for day, score in buy:
            buys.setdefault(day, []).append((stock.column, score))

    equity, exposure, trades, counts = simulate(market, buys, overbought, neutral, stop_loss)
    benchmark_return = kospi_close.pct_change(fill_method=None).fillna(0)
    metrics = performance(equity, exposure, benchmark_return, trades)
    return metrics, trades, equity, exposure, counts


def load_or_prepare(refresh_cache: bool) -> tuple[SimulationInputs, list[StockFeatures], dict[int, set[int]], dict[int, set[int]], pd.Series]:
    if CACHE.exists() and not refresh_cache:
        print(f"준비 캐시 읽는 중: {CACHE}", flush=True)
        with CACHE.open("rb") as file:
            market, features, overbought, neutral, kospi_close = pickle.load(file)
        print(f"캐시 로드 완료: 대상 {len(features):,}종목", flush=True)
        return market, features, overbought, neutral, kospi_close

    close, open_, volume, entry_eligible, kospi_close = load_data()
    print(f"{len(close):,}일 × {close.shape[1]:,}종목", flush=True)
    print("공통 지표·피봇 계산 (전체 실행에서 1회)...", flush=True)
    market = prepare_market(close, open_, entry_eligible)
    features, overbought, neutral = prepare_features(close, volume, kospi_close)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"준비 캐시 저장 중: {CACHE}", flush=True)
    with CACHE.open("wb") as file:
        pickle.dump((market, features, overbought, neutral, kospi_close), file, protocol=pickle.HIGHEST_PROTOCOL)
    return market, features, overbought, neutral, kospi_close



def period_rows(
    base: dict[str, object],
    equity: pd.Series,
    exposure: pd.Series,
    kospi_close: pd.Series,
    trades: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    benchmark_return = kospi_close.pct_change(fill_method=None).fillna(0)
    for name, start, end in PERIODS:
        period_equity = equity.loc[start:end]
        if len(period_equity) < 2:
            continue
        period_exposure = exposure.reindex(period_equity.index).fillna(0)
        if trades.empty:
            period_trades = trades
        else:
            exits = pd.to_datetime(trades["exit_date"])
            period_trades = trades.loc[(exits >= pd.Timestamp(start)) & (exits <= pd.Timestamp(end))]
        metrics = performance(
            period_equity / period_equity.iloc[0],
            period_exposure,
            benchmark_return.reindex(period_equity.index).fillna(0),
            period_trades,
        )
        rows.append({
            **base,
            "period": name,
            "start": start,
            "end": end,
            **metrics,
            "period_trades": len(period_trades),
        })
    return rows

def score_result(row: dict[str, object]) -> float:
    cagr = float(row["cagr"])
    mdd = abs(float(row["mdd"]))
    trade_pf = float(row["trade_pf"]) if np.isfinite(float(row["trade_pf"])) else 0.0
    filled = int(row["filled"])
    exposure = float(row["exposure"])
    return cagr * 100 + trade_pf * 2 + min(filled, 200) / 100 - mdd * 10 - exposure



def slice_market_and_features(
    market: SimulationInputs,
    features: list[StockFeatures],
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    start: str,
    end: str,
) -> tuple[SimulationInputs, list[StockFeatures], dict[int, set[int]], dict[int, set[int]], slice]:
    dates = market.dates
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        empty = SimulationInputs(
            dates=pd.DatetimeIndex([]),
            tickers=market.tickers,
            close=market.close[:0],
            open=market.open[:0],
            marked_close=market.marked_close[:0],
            entry_eligible=market.entry_eligible[:0],
        )
        return empty, [], {}, {}, slice(0, 0)
    start_i, end_i = int(indices[0]), int(indices[-1]) + 1
    window = slice(start_i, end_i)
    sliced_market = SimulationInputs(
        dates=pd.DatetimeIndex(dates[window]),
        tickers=market.tickers,
        close=market.close[window],
        open=market.open[window],
        marked_close=market.marked_close[window],
        entry_eligible=market.entry_eligible[window],
    )
    sliced_features: list[StockFeatures] = []
    for stock in features:
        lows = [(i - start_i, c - start_i) for i, c in stock.lows if start_i <= i and c < end_i]
        highs = [(i - start_i, c - start_i) for i, c in stock.highs if start_i <= i and c < end_i]
        sliced_features.append(
            StockFeatures(
                column=stock.column,
                price=stock.price[window],
                strength=stock.strength[window],
                volume=stock.volume[window],
                relative_strength=stock.relative_strength[window],
                lows=lows,
                highs=highs,
            )
        )

    def shift_signals(signals: dict[int, set[int]]) -> dict[int, set[int]]:
        return {day - start_i: set(names) for day, names in signals.items() if start_i <= day < end_i}

    return sliced_market, sliced_features, shift_signals(overbought), shift_signals(neutral), window


def independent_period_rows(
    summary: pd.DataFrame,
    market: SimulationInputs,
    features: list[StockFeatures],
    overbought: dict[int, set[int]],
    neutral: dict[int, set[int]],
    kospi_close: pd.Series,
    top_n: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, selected in enumerate(summary.head(top_n).itertuples(index=False), 1):
        for period, start, end in PERIODS:
            period_market, period_features, period_ob, period_neutral, window = slice_market_and_features(
                market, features, overbought, neutral, start, end
            )
            if len(period_market.dates) < 2:
                continue
            metrics, trades, _equity, _exposure, counts = run_one(
                period_market,
                period_features,
                period_ob,
                period_neutral,
                kospi_close.iloc[window],
                int(selected.reentry_rsi),
                int(selected.confirm_days),
                float(selected.volume_factor),
                bool(selected.use_relative_strength),
                None if pd.isna(selected.stop_loss) else float(selected.stop_loss),
            )
            rows.append({
                "rank_full": rank,
                "period": period,
                "start": start,
                "end": end,
                "reentry_rsi": int(selected.reentry_rsi),
                "confirm_days": int(selected.confirm_days),
                "volume_factor": float(selected.volume_factor),
                "use_relative_strength": bool(selected.use_relative_strength),
                **metrics,
                **counts,
                "trades": len(trades),
                "pass_alpha_gate": metrics["cagr"] > 0 and metrics["trade_pf"] > 1.2 and metrics["t_exposure_matched"] > 1.0,
            })
    return rows

def main() -> None:
    started = perf_counter()
    refresh_cache = "--refresh-cache" in sys.argv
    print("SeedForge 006 RSI 다이버전스 백테스트 시작", flush=True)
    market, features, overbought, neutral, kospi_close = load_or_prepare(refresh_cache)
    print(
        f"대상 {len(features):,}종목, 전처리/캐시 {perf_counter() - started:.1f}초, "
        "조합별 신호/시뮬레이션 시작",
        flush=True,
    )
    print("RSI 대기 거래량 RS  손절   CAGR    MDD    노출   t(노출조정) 승률  손익비 거래PF 거래수 놓침")
    output = D / "results"
    output.mkdir(parents=True, exist_ok=True)
    combination = 0
    summaries: list[dict[str, object]] = []
    period_summaries: list[dict[str, object]] = []
    for reentry in (35, 40, 45):
        for wait in (5, 10, 15):
            for volume_factor in (1.0, 1.2):
                for use_rs in (False, True):
                    for stop_loss in (None, -0.30):
                        combination += 1
                        combination_started = perf_counter()
                        metrics, trades, equity, exposure, counts = run_one(
                            market,
                            features,
                            overbought,
                            neutral,
                            kospi_close,
                            reentry,
                            wait,
                            volume_factor,
                            use_rs,
                            stop_loss,
                        )
                        stop_tag = "nostop" if stop_loss is None else "stop30"
                        tag = f"r{reentry}_w{wait}_v{volume_factor:.1f}_rs{int(use_rs)}_{stop_tag}"
                        trades.to_json(
                            output / f"trades_{tag}.json",
                            orient="records",
                            date_format="iso",
                            force_ascii=False,
                        )
                        equity.to_csv(output / f"equity_{tag}.csv", header=True)
                        missed = counts["skipped"] / max(1, counts["signals"])
                        row = {
                            "reentry_rsi": reentry,
                            "confirm_days": wait,
                            "volume_factor": volume_factor,
                            "use_relative_strength": use_rs,
                            "stop_loss": np.nan if stop_loss is None else stop_loss,
                            **metrics,
                            **counts,
                            "missed": missed,
                        }
                        row["score"] = score_result(row)
                        summaries.append(row)
                        period_summaries.extend(period_rows(row, equity, exposure, kospi_close, trades))
                        print(
                            f"[{combination:>2}/72] "
                            f"{reentry:<3} {wait:<4} {volume_factor:<4.1f} {'O' if use_rs else '-':<2} {stop_tag:<6} "
                            f"{metrics['cagr']:>7.2%} {metrics['mdd']:>7.1%} "
                            f"{metrics['exposure']:>6.1%} {metrics['t_exposure_matched']:>10.2f} "
                            f"{metrics['win_rate']:>5.1%} {metrics['payoff']:>6.2f} "
                            f"{metrics['trade_pf']:>6.2f} {counts['filled']:>5} {missed:>5.0%} "
                            f"{perf_counter() - combination_started:>5.1f}초",
                            flush=True,
                        )
    summary = pd.DataFrame(summaries).sort_values("score", ascending=False)
    summary.to_csv(output / "summary_seedforge_006.csv", index=False, encoding="utf-8-sig")
    period_summary = pd.DataFrame(period_summaries)
    period_summary.to_csv(output / "periods_seedforge_006.csv", index=False, encoding="utf-8-sig")
    print("\n상위 5개 조합(점수순)", flush=True)
    for rank, row in enumerate(summary.head(5).itertuples(index=False), 1):
        print(
            f"#{rank} RSI {row.reentry_rsi} / 대기 {row.confirm_days} / "
            f"거래량 {row.volume_factor:.1f} / RS {'O' if row.use_relative_strength else '-'} / "
            f"손절 {'없음' if pd.isna(row.stop_loss) else f'{row.stop_loss:.0%}'} | "
            f"CAGR {row.cagr:.2%}, MDD {row.mdd:.1%}, PF {row.trade_pf:.2f}, "
            f"거래수 {row.filled}, 점수 {row.score:.2f}",
            flush=True,
        )
    print(f"요약 저장: {output / 'summary_seedforge_006.csv'}", flush=True)
    print(f"기간분리 저장: {output / 'periods_seedforge_006.csv'}", flush=True)
    independent = pd.DataFrame(
        independent_period_rows(summary, market, features, overbought, neutral, kospi_close)
    )
    independent.to_csv(output / "walkforward_seedforge_006.csv", index=False, encoding="utf-8-sig")
    print(f"독립 기간 재시뮬레이션 저장: {output / 'walkforward_seedforge_006.csv'}", flush=True)
    best = summary.iloc[0]
    best_periods = period_summary.loc[
        (period_summary["reentry_rsi"] == best["reentry_rsi"])
        & (period_summary["confirm_days"] == best["confirm_days"])
        & (period_summary["volume_factor"] == best["volume_factor"])
        & (period_summary["use_relative_strength"] == best["use_relative_strength"])
    ]
    print("\n1위 조합 기간분리", flush=True)
    for row in best_periods.itertuples(index=False):
        print(
            f"{row.period}: CAGR {row.cagr:.2%}, MDD {row.mdd:.1%}, "
            f"t {row.t_exposure_matched:.2f}, PF {row.trade_pf:.2f}, 거래 {row.period_trades}",
            flush=True,
        )
    print(f"전체 실행시간 {perf_counter() - started:.1f}초", flush=True)


if __name__ == "__main__":
    main()
