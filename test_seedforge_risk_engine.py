import unittest

import numpy as np
import pandas as pd

from seedforge_risk_engine import (
    AssetStatus,
    CostModel,
    MarketFixture,
    PortfolioPolicy,
    WithdrawalRiskPolicy,
    deterministic_ranking,
    simulate_audited_portfolio,
)


def market(open_values, close_values=None, status=None, eligible=None, recovery=None):
    opens = np.asarray(open_values, dtype=float)
    closes = opens.copy() if close_values is None else np.asarray(close_values, dtype=float)
    shape = opens.shape
    return MarketFixture(
        dates=pd.date_range("2020-01-01", periods=shape[0], freq="D"),
        tickers=tuple(f"T{i}" for i in range(shape[1])),
        open=opens,
        close=closes,
        eligible=np.ones(shape, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool),
        status=(np.full(shape, AssetStatus.ACTIVE, dtype=np.int8)
                if status is None else np.asarray(status, dtype=np.int8)),
        delisting_recovery=(np.full(shape, np.nan)
                            if recovery is None else np.asarray(recovery, dtype=float)),
    )


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.zero_cost = CostModel(0, 0, 0, 0)

    def test_ties_use_ticker(self):
        self.assertEqual(deterministic_ranking([1, 1, 2], ["B", "A", "C"]), (2, 1, 0))

    def test_half_exit_remains_half_on_rebalance_day(self):
        fixture = market([[10], [10], [10]])
        result = simulate_audited_portfolio(
            fixture, {0: (0,), 1: (0,)}, {}, {0: {0}},
            PortfolioPolicy(1, 1), costs=self.zero_cost,
        )
        exits = result.ledger.loc[result.ledger["side"].eq("sell")]
        half = exits.loc[exits["reason"].eq("overbought_half"), "quantity"].iloc[0]
        self.assertAlmostEqual(half, 0.05)
        self.assertFalse((exits["reason"] == "rebalance_exit").any())

    def test_delisting_uses_explicit_recovery(self):
        statuses = [[1], [3]]
        recovery = [[np.nan], [2.0]]
        fixture = market([[10], [np.nan]], close_values=[[10], [np.nan]],
                         status=statuses, recovery=recovery)
        result = simulate_audited_portfolio(
            fixture, {0: (0,)}, {}, {}, PortfolioPolicy(1, 1), costs=self.zero_cost,
        )
        self.assertEqual(result.forced_delisting_exits, 1)
        self.assertAlmostEqual(result.final_account_nav, 0.2)
        self.assertEqual(result.unresolved_positions, 0)

    def test_suspension_is_unresolved_not_fake_liquidation(self):
        fixture = market([[10], [np.nan]], close_values=[[10], [np.nan]], status=[[1], [2]])
        result = simulate_audited_portfolio(
            fixture, {0: (0,)}, {}, {}, PortfolioPolicy(1, 1), costs=self.zero_cost,
        )
        self.assertEqual(result.unresolved_positions, 1)
        self.assertAlmostEqual(result.final_account_nav, 1.0)

    def test_final_liquidation_charges_sell_cost(self):
        costs = CostModel(commission_rate=0.01, sell_tax_rate=0.02,
                          buy_slippage_rate=0, sell_slippage_rate=0)
        fixture = market([[10], [10]])
        result = simulate_audited_portfolio(
            fixture, {0: (0,)}, {}, {}, PortfolioPolicy(1, 1), costs=costs,
        )
        self.assertEqual(result.unresolved_positions, 0)
        self.assertGreater(result.fees, 0)
        self.assertGreater(result.taxes, 0)
        self.assertLess(result.final_account_nav, 1.0)

    def test_doubling_withdraws_half_without_counting_a_loss(self):
        fixture = market([[10], [20], [20]])
        result = simulate_audited_portfolio(
            fixture, {0: (0,), 1: (0,), 2: (0,)}, {}, {}, PortfolioPolicy(1, 3),
            withdrawal=WithdrawalRiskPolicy(), costs=self.zero_cost,
        )
        self.assertEqual(len(result.withdrawals), 1)
        self.assertAlmostEqual(result.cumulative_withdrawals, 1.0)
        self.assertAlmostEqual(result.final_account_nav, 1.0)
        self.assertAlmostEqual(result.final_total_wealth, 2.0)
        self.assertTrue(bool(result.daily.iloc[-1]["aggressive"]))

    def test_aggression_reduces_size_after_withdrawal(self):
        fixture = market([[10, 10], [20, 20], [20, 20], [20, 20]])
        result = simulate_audited_portfolio(
            fixture, {0: (0, 1), 1: (0, 1), 2: (0, 1)}, {}, {},
            PortfolioPolicy(2, 3), WithdrawalRiskPolicy(aggressive_size_multiplier=0.5),
            self.zero_cost,
        )
        self.assertTrue(bool(result.daily.iloc[-1]["aggressive"]))
        self.assertLessEqual(int(result.daily.iloc[-1]["positions"]), 1)


if __name__ == "__main__":
    unittest.main()

