"""Auditable portfolio/risk primitives for the post-033 SeedForge engine.

This module does not reinterpret the legacy 033 CSV.  It provides a small,
deterministic engine for hand-calculated fixtures before any expensive rerun.
Withdrawals are external cash flows: performance is measured as account NAV
plus cumulative withdrawals, never from the reduced account NAV alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class AssetStatus(IntEnum):
    """Point-in-time trading state used by the simulator."""

    ACTIVE = 1
    SUSPENDED = 2
    DELISTED = 3


@dataclass(frozen=True)
class CostModel:
    """Direction-aware Korean-equity cost model.

    Defaults sum to a 0.58% round trip: commission 0.015% on each side,
    sell tax 0.15%, and slippage 0.20% on each side.
    """

    commission_rate: float = 0.00015
    sell_tax_rate: float = 0.0015
    buy_slippage_rate: float = 0.002
    sell_slippage_rate: float = 0.002

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0 or value >= 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")

    def buy_cash(self, quote: float, quantity: float) -> tuple[float, float, float]:
        execution = quote * (1 + self.buy_slippage_rate)
        gross = execution * quantity
        fee = gross * self.commission_rate
        return gross + fee, fee, gross - quote * quantity

    def sell_cash(self, quote: float, quantity: float) -> tuple[float, float, float, float]:
        execution = quote * (1 - self.sell_slippage_rate)
        gross = execution * quantity
        fee = gross * self.commission_rate
        tax = gross * self.sell_tax_rate
        return gross - fee - tax, fee, tax, quote * quantity - gross


@dataclass(frozen=True)
class WithdrawalRiskPolicy:
    """Recover capital whenever the account doubles, then increase aggression.

    At ``trigger_multiple=2`` and ``withdraw_fraction=0.5``, half of the
    liquidatable account is withdrawn when it reaches twice its current risk
    basis.  The post-withdrawal NAV becomes the next risk basis.  This allows
    repeated, unambiguous withdrawals without counting withdrawals as losses.
    """

    trigger_multiple: float = 2.0
    withdraw_fraction: float = 0.5
    activate_aggression: bool = True
    aggressive_size_multiplier: float = 0.5
    aggressive_rebalance_months: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.trigger_multiple) or self.trigger_multiple <= 1:
            raise ValueError("trigger_multiple must be greater than 1")
        if not np.isfinite(self.withdraw_fraction) or not 0 < self.withdraw_fraction < 1:
            raise ValueError("withdraw_fraction must be in (0, 1)")
        if not 0 < self.aggressive_size_multiplier <= 1:
            raise ValueError("aggressive_size_multiplier must be in (0, 1]")
        if self.aggressive_rebalance_months < 1:
            raise ValueError("aggressive_rebalance_months must be positive")


@dataclass(frozen=True)
class PortfolioPolicy:
    size: int
    rebalance_months: int
    exit_rank_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.size < 1 or self.rebalance_months < 1 or self.exit_rank_multiplier < 1:
            raise ValueError("invalid portfolio policy")


@dataclass(frozen=True)
class MarketFixture:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    open: np.ndarray
    close: np.ndarray
    eligible: np.ndarray
    status: np.ndarray
    delisting_recovery: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.tickers))
        for name in ("open", "close", "eligible", "status", "delisting_recovery"):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} shape mismatch")
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("dates must be unique and increasing")
        allowed = {int(item) for item in AssetStatus}
        if not set(np.unique(self.status)).issubset(allowed):
            raise ValueError("unknown asset status")


@dataclass
class EngineResult:
    ledger: pd.DataFrame
    daily: pd.DataFrame
    withdrawals: pd.DataFrame
    final_account_nav: float
    cumulative_withdrawals: float
    final_total_wealth: float
    fees: float
    taxes: float
    slippage: float
    gross_traded: float
    forced_delisting_exits: int
    unresolved_positions: int


def deterministic_ranking(scores: Sequence[float], tickers: Sequence[str]) -> tuple[int, ...]:
    """Sort descending by score and ascending by ticker for exact ties."""

    candidates = [index for index, value in enumerate(scores) if np.isfinite(value)]
    return tuple(sorted(candidates, key=lambda index: (-float(scores[index]), str(tickers[index]))))


def simulate_audited_portfolio(
    market: MarketFixture,
    rankings: Mapping[int, Sequence[int]],
    neutral_signals: Mapping[int, set[int]] | None,
    overbought_signals: Mapping[int, set[int]] | None,
    portfolio: PortfolioPolicy,
    withdrawal: WithdrawalRiskPolicy | None = None,
    costs: CostModel = CostModel(),
    initial_capital: float = 1.0,
) -> EngineResult:
    """Simulate deterministic next-open orders with explicit external withdrawals."""

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    neutral_signals = neutral_signals or {}
    overbought_signals = overbought_signals or {}
    cash = float(initial_capital)
    shares: dict[int, float] = {}
    last_mark = np.full(len(market.tickers), np.nan, dtype=float)
    risk_basis = float(initial_capital)
    cumulative_withdrawals = fees = taxes = slippage = gross_traded = 0.0
    forced_delisting_exits = 0
    aggressive = False
    rebalance_number = 0
    ledger: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    withdrawals: list[dict[str, object]] = []

    def mark(day: int, stock: int) -> float:
        close = float(market.close[day, stock])
        if np.isfinite(close) and close >= 0:
            last_mark[stock] = close
        return float(last_mark[stock])

    def account_nav(day: int, prefer_open: bool = False) -> float:
        value = cash
        for stock, quantity in shares.items():
            quote = float(market.open[day, stock]) if prefer_open else np.nan
            if not np.isfinite(quote) or quote <= 0:
                quote = mark(day, stock)
            if np.isfinite(quote) and quote >= 0:
                value += quantity * quote
        return float(value)

    def sell(day: int, stock: int, fraction: float, reason: str, quote: float | None = None) -> bool:
        nonlocal cash, fees, taxes, slippage, gross_traded
        held = shares.get(stock, 0.0)
        price = float(market.open[day, stock]) if quote is None else float(quote)
        if held <= 0 or not np.isfinite(price) or price < 0:
            ledger.append({"date": market.dates[day], "ticker": market.tickers[stock],
                           "side": "sell_rejected", "reason": reason, "quantity": 0.0,
                           "cash_after": cash})
            return False
        quantity = held * min(max(fraction, 0.0), 1.0)
        proceeds, fee, tax, slip = costs.sell_cash(price, quantity)
        cash += proceeds
        fees += fee
        taxes += tax
        slippage += slip
        gross_traded += price * quantity
        remaining = held - quantity
        if remaining <= held * 1e-12:
            shares.pop(stock, None)
        else:
            shares[stock] = remaining
        ledger.append({"date": market.dates[day], "ticker": market.tickers[stock],
                       "side": "sell", "reason": reason, "quantity": quantity,
                       "quote": price, "fee": fee, "tax": tax, "slippage": slip,
                       "cash_after": cash})
        return True

    def buy(day: int, stock: int, desired_cash: float, reason: str) -> bool:
        nonlocal cash, fees, slippage, gross_traded
        price = float(market.open[day, stock])
        if (desired_cash <= 0 or not market.eligible[day, stock]
                or market.status[day, stock] != AssetStatus.ACTIVE
                or not np.isfinite(price) or price <= 0):
            return False
        unit_cash, _, _ = costs.buy_cash(price, 1.0)
        quantity = min(desired_cash, cash) / unit_cash
        if quantity <= 0:
            return False
        spent, fee, slip = costs.buy_cash(price, quantity)
        shares[stock] = shares.get(stock, 0.0) + quantity
        cash -= spent
        fees += fee
        slippage += slip
        gross_traded += price * quantity
        ledger.append({"date": market.dates[day], "ticker": market.tickers[stock],
                       "side": "buy", "reason": reason, "quantity": quantity,
                       "quote": price, "fee": fee, "tax": 0.0, "slippage": slip,
                       "cash_after": cash})
        return True

    def raise_cash(day: int, target_cash: float) -> None:
        if target_cash <= cash or not shares:
            return
        liquid = [stock for stock in sorted(shares, key=lambda item: market.tickers[item])
                  if market.status[day, stock] == AssetStatus.ACTIVE
                  and np.isfinite(market.open[day, stock]) and market.open[day, stock] > 0]
        for stock in liquid:
            needed = max(target_cash - cash, 0.0)
            if needed <= 1e-15:
                break
            quote = float(market.open[day, stock])
            net_per_share, _, _, _ = costs.sell_cash(quote, 1.0)
            quantity = min(shares[stock], needed / max(net_per_share, 1e-15))
            sell(day, stock, quantity / shares[stock], "withdrawal_funding")

    for day in range(len(market.dates)):
        for stock in range(len(market.tickers)):
            mark(day, stock)

        # Delisting is distinct from suspension and has an explicit recovery value.
        for stock in tuple(shares):
            if market.status[day, stock] == AssetStatus.DELISTED:
                recovery = float(market.delisting_recovery[day, stock])
                if not np.isfinite(recovery) or recovery < 0:
                    recovery = 0.0
                if sell(day, stock, 1.0, "forced_delisting", quote=recovery):
                    forced_delisting_exits += 1

        # The threshold is checked on liquidatable opening NAV before new signals.
        if withdrawal is not None:
            nav_before = account_nav(day, prefer_open=True)
            if nav_before >= risk_basis * withdrawal.trigger_multiple:
                requested = nav_before * withdrawal.withdraw_fraction
                raise_cash(day, requested)
                paid = min(requested, cash)
                cash -= paid
                cumulative_withdrawals += paid
                risk_basis = account_nav(day, prefer_open=True)
                aggressive = aggressive or withdrawal.activate_aggression
                withdrawals.append({"date": market.dates[day], "nav_before": nav_before,
                                    "requested": requested, "paid": paid,
                                    "nav_after": risk_basis,
                                    "cumulative_withdrawals": cumulative_withdrawals})

        blocked_from_buy: set[int] = set()
        partial_retention: set[int] = set()
        if day > 0:
            neutral_today = neutral_signals.get(day - 1, set())
            overbought_today = overbought_signals.get(day - 1, set()) - neutral_today
            for stock in tuple(shares):
                if stock in neutral_today and sell(day, stock, 1.0, "neutral_divergence"):
                    blocked_from_buy.add(stock)
                elif stock in overbought_today and sell(day, stock, 0.5, "overbought_half"):
                    blocked_from_buy.add(stock)
                    partial_retention.add(stock)

        ranking = tuple(rankings.get(day, ()))
        if ranking:
            months = (withdrawal.aggressive_rebalance_months
                      if aggressive and withdrawal is not None else portfolio.rebalance_months)
            execute = rebalance_number % months == 0
            rebalance_number += 1
            if execute:
                size = portfolio.size
                if aggressive and withdrawal is not None:
                    size = max(1, int(np.ceil(size * withdrawal.aggressive_size_multiplier)))
                exit_rank = max(size, int(np.ceil(size * portfolio.exit_rank_multiplier)))
                allowed = set(ranking[:exit_rank]) - blocked_from_buy
                retained = [stock for stock in shares if stock in allowed or stock in partial_retention]
                target = retained[:size]
                for stock in ranking:
                    if len(target) >= size:
                        break
                    if stock not in target and stock not in blocked_from_buy:
                        target.append(stock)
                target_set = set(target)
                for stock in tuple(shares):
                    if stock not in target_set:
                        sell(day, stock, 1.0, "rebalance_exit")
                nav = account_nav(day, prefer_open=True)
                target_value = nav / max(size, 1)
                for stock in target:
                    price = float(market.open[day, stock])
                    if np.isfinite(price) and price > 0:
                        value = shares.get(stock, 0.0) * price
                        if value > target_value:
                            sell(day, stock, (value - target_value) / value, "rebalance_trim")
                for stock in target:
                    price = float(market.open[day, stock])
                    if stock not in blocked_from_buy and np.isfinite(price) and price > 0:
                        value = shares.get(stock, 0.0) * price
                        if value < target_value:
                            buy(day, stock, target_value - value, "rebalance_buy")

        nav = account_nav(day)
        daily.append({"date": market.dates[day], "account_nav": nav,
                      "cumulative_withdrawals": cumulative_withdrawals,
                      "total_wealth": nav + cumulative_withdrawals,
                      "risk_basis": risk_basis, "aggressive": aggressive,
                      "positions": len(shares), "cash": cash})

    # Conservative final liquidation. Suspended assets remain explicitly unresolved.
    final_day = len(market.dates) - 1
    for stock in tuple(shares):
        quote = float(market.open[final_day, stock])
        if market.status[final_day, stock] == AssetStatus.ACTIVE and np.isfinite(quote) and quote > 0:
            sell(final_day, stock, 1.0, "end_of_test")
    unresolved_positions = len(shares)
    final_nav = account_nav(final_day)
    if daily:
        daily[-1].update({"account_nav": final_nav,
                          "total_wealth": final_nav + cumulative_withdrawals,
                          "positions": len(shares), "cash": cash})
    return EngineResult(
        ledger=pd.DataFrame(ledger), daily=pd.DataFrame(daily),
        withdrawals=pd.DataFrame(withdrawals), final_account_nav=final_nav,
        cumulative_withdrawals=cumulative_withdrawals,
        final_total_wealth=final_nav + cumulative_withdrawals,
        fees=fees, taxes=taxes, slippage=slippage, gross_traded=gross_traded,
        forced_delisting_exits=forced_delisting_exits,
        unresolved_positions=unresolved_positions,
    )
