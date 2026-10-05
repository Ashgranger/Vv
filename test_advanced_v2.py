"""Comprehensive Verification Suite for Level 9+ Institutional Market Maker Engine.

Tests:
1. Hawkes Point Process Engine (Intensity, Decay, Branching Ratio, Cascade Detection, Skew)
2. Cartea-Jaimungal & Guéant-Tapia-Manziadi Stochastic Optimal Control (Reservation Price, Asymmetric Spreads, Volatility Response)
3. Active Delta Hedger (Normal Threshold, Cascade Emergency, Tranche Sizing, Hedge Execution Accounting)
4. Full End-to-End Integrated Simulation with Hawkes & Optimal Control Active
"""
import asyncio
import math
import os
import sys
import unittest
from decimal import Decimal as D

import sim
from config import Config
from hawkes import HawkesProcessEngine
from hedger import DeltaHedger, HedgeSignal
from optimal_control import CarteaJaimungalEngine, OptimalQuotes
from utils import BUY, SELL, ZERO, fmt


class TestAdvancedMarketMakerV2(unittest.IsolatedAsyncioTestCase):

    def test_01_hawkes_intensity_and_decay(self):
        """Test Hawkes continuous-time intensity calculation and exponential decay."""
        engine = HawkesProcessEngine(decay_beta=2.0)
        t0 = 100.0

        # Initial baseline intensity
        base_buy = engine.get_intensity("buy", t0)
        self.assertAlmostEqual(base_buy, 1.0, places=3)

        # Record aggressive buy trade at t0
        engine.record_event("buy", t0, size=2.0)
        int_t0 = engine.get_intensity("buy", t0)
        self.assertGreater(int_t0, base_buy, "Intensity should jump upon trade arrival")

        # Advance time by 0.5s -> intensity decays towards baseline
        t1 = t0 + 0.5
        int_t1 = engine.get_intensity("buy", t1)
        self.assertLess(int_t1, int_t0, "Intensity should decay over time")
        self.assertGreater(int_t1, base_buy, "Intensity should still be elevated at 0.5s")

        # Advance time by 5.0s -> nearly returned to baseline
        t2 = t0 + 5.0
        int_t2 = engine.get_intensity("buy", t2)
        self.assertAlmostEqual(int_t2, base_buy, places=1)
        print("✓ test_01_hawkes_intensity_and_decay passed: Intensity jump and exponential decay verified.")

    def test_02_hawkes_branching_ratio_and_criticality(self):
        """Test Hawkes branching ratio (spectral radius) computation."""
        # Standard stable engine: rho < 1.0
        engine = HawkesProcessEngine(decay_beta=2.0)
        rho = engine.get_branching_ratio()
        self.assertGreater(rho, 0.0)
        self.assertLess(rho, 1.0, "Subcritical branching ratio should be < 1.0")

        # Highly excited engine: rho >= 1.0
        critical_alpha = {
            ("buy", "buy"): 2.5,
            ("sell", "sell"): 2.5,
            ("cancel", "cancel"): 1.5,
        }
        critical_engine = HawkesProcessEngine(decay_beta=2.0, alpha_matrix=critical_alpha)
        rho_crit = critical_engine.get_branching_ratio()
        self.assertGreaterEqual(rho_crit, 1.0, "Supercritical branching ratio should be >= 1.0")
        print("✓ test_02_hawkes_branching_ratio_and_criticality passed: Branching matrix spectral radius verified.")

    def test_03_hawkes_cascade_detection_and_quote_skew(self):
        """Test toxic cascade detection and adaptive quote widening."""
        engine = HawkesProcessEngine(decay_beta=2.0, cascade_threshold_multiplier=4.0)
        t = 200.0

        # Normal condition -> no cascade
        self.assertFalse(engine.is_cascade_active("buy", t))
        self.assertEqual(engine.get_quote_skew_bps("buy", t), D("0.0"))

        # Burst of 5 aggressive sell trades in 0.2s -> triggers cascade on buy side
        for i in range(5):
            engine.record_event("sell", t + i * 0.04, size=3.0)

        t_burst = t + 0.2
        self.assertTrue(engine.is_cascade_active("buy", t_burst), "Sell burst must activate cascade against BUY quotes")
        self.assertFalse(engine.is_cascade_active("sell", t_burst), "Buy quotes should not be in cascade from sells")

        skew_bps = engine.get_quote_skew_bps("buy", t_burst)
        self.assertGreater(skew_bps, D("1.0"), "Quote widening skew should activate during cascade")
        print("✓ test_03_hawkes_cascade_detection_and_quote_skew passed: Burst sell cascade and quote skew verified.")

    def test_04_cartea_jaimungal_reservation_price(self):
        """Test Cartea-Jaimungal closed-form reservation price with inventory penalty and alpha."""
        cj = CarteaJaimungalEngine(gamma=0.15, kappa=1.5, terminal_horizon_s=300.0)
        mid = D("80000.0")

        # Flat inventory -> res price == fair value (zero alpha)
        res_flat = cj.compute_reservation_price(mid, position_norm=0.0, vol_bps=10.0, alpha_bps=0.0)
        self.assertEqual(res_flat, mid)

        # Long inventory (q = +0.5) -> res price < mid (wants to sell)
        res_long = cj.compute_reservation_price(mid, position_norm=0.5, vol_bps=10.0, alpha_bps=0.0)
        self.assertLess(res_long, mid, "Long inventory must skew reservation price below mid")

        # Short inventory (q = -0.5) -> res price > mid (wants to buy)
        res_short = cj.compute_reservation_price(mid, position_norm=-0.5, vol_bps=10.0, alpha_bps=0.0)
        self.assertGreater(res_short, mid, "Short inventory must skew reservation price above mid")

        # Positive alpha drift -> shifts reservation price higher
        res_alpha = cj.compute_reservation_price(mid, position_norm=0.0, vol_bps=10.0, alpha_bps=5.0)
        self.assertGreater(res_alpha, mid, "Positive alpha must increase reservation price")
        print("✓ test_04_cartea_jaimungal_reservation_price passed: Inventory discount and alpha drift verified.")

    def test_05_cartea_jaimungal_optimal_half_spreads(self):
        """Test Cartea-Jaimungal closed-form optimal half-spreads."""
        cj = CarteaJaimungalEngine(gamma=0.15, kappa=1.5, arrival_intensity=5.0)

        # Flat inventory: symmetric spreads
        delta_b_flat, delta_a_flat = cj.compute_optimal_half_spreads(0.0, vol_bps=10.0)
        self.assertAlmostEqual(float(delta_b_flat), float(delta_a_flat), places=1)

        # Higher volatility -> spreads widen
        delta_b_high_vol, delta_a_high_vol = cj.compute_optimal_half_spreads(0.0, vol_bps=30.0)
        self.assertGreater(delta_b_high_vol, delta_b_flat, "Spreads must widen with higher volatility")

        # Long inventory (q = +0.7): Ask spread narrows (to unwind), Bid spread widens (to deter buying)
        delta_b_long, delta_a_long = cj.compute_optimal_half_spreads(0.7, vol_bps=10.0)
        self.assertGreater(delta_b_long, delta_a_long, "When long, bid spread must be wider than ask spread")
        print("✓ test_05_cartea_jaimungal_optimal_half_spreads passed: Volatility and inventory asymmetric spreads verified.")

    def test_06_delta_hedger_thresholds_and_tranching(self):
        """Test Active Delta Hedger triggering, direction, and tranche sizing."""
        hedger = DeltaHedger(
            max_position_usd=D("100.0"),
            hedge_trigger_ratio=0.75,   # > $75
            target_hedge_ratio=0.35,    # hedge down to $35
            cascade_trigger_ratio=0.45, # > $45 during cascade
            max_tranche_usd=D("25.0"),
            min_hedge_notional=D("5.0"),
        )
        mid = D("80000.0")

        # Normal inventory $50 (50%) -> No hedge
        sig1 = hedger.evaluate(position_usd=D("50.0"), current_mid=mid, is_hawkes_cascade=False, now=10.0)
        self.assertFalse(sig1.should_hedge)

        # Long inventory $85 (85% > 75%) -> Must hedge SELL
        sig2 = hedger.evaluate(position_usd=D("85.0"), current_mid=mid, is_hawkes_cascade=False, now=11.0)
        self.assertTrue(sig2.should_hedge)
        self.assertEqual(sig2.side, SELL)
        # Target is 35 -> excess is 85 - 35 = 50. Max tranche is 25.
        expected_qty = D("25.0") / mid
        self.assertAlmostEqual(float(sig2.qty), float(expected_qty), places=6)

        # Short inventory -$55 during Hawkes cascade (55% > 45% cascade trigger) -> Must hedge BUY
        sig3 = hedger.evaluate(position_usd=D("-55.0"), current_mid=mid, is_hawkes_cascade=True, now=12.0)
        self.assertTrue(sig3.should_hedge)
        self.assertEqual(sig3.side, BUY)
        self.assertEqual(sig3.urgency, "EMERGENCY_CASCADE")

        # Record execution
        hedger.record_hedge_execution(sig3.qty, mid, now=12.0)
        self.assertEqual(hedger.total_hedge_events, 1)
        self.assertGreater(hedger.total_hedged_notional, D(0))
        print("✓ test_06_delta_hedger_thresholds_and_tranching passed: Normal & cascade triggers with tranche sizing verified.")

    async def test_07_full_system_integration_with_hawkes_and_optimal_control(self):
        """Test full Arcus Level 9+ Bot simulation with Hawkes and Optimal Control actively quoting."""
        sim.MKT.funding_rate = D("0")
        bot, s, clock = sim.make(
            MARKET="BTC-USD",
            QUOTE_OUTSIDE_RTH="1",
            ENABLE_HAWKES="1",
            ENABLE_OPTIMAL_CONTROL="1",
            ENABLE_DELTA_HEDGING="1",
            ORDER_USD="20",
            MAX_POSITION_USD="100",
            EXTRA_LEVELS="1",
        )
        # Advance clock and step market
        await sim.step(bot, s, clock, "80000.0", "80020.0")

        # Verify active quotes
        buy_orders = bot.om.side_orders(BUY)
        sell_orders = bot.om.side_orders(SELL)
        self.assertGreaterEqual(len(buy_orders), 1, "Buy orders should be resting")
        self.assertGreaterEqual(len(sell_orders), 1, "Sell orders should be resting")

        # Inject aggressive burst trade flow
        for i in range(4):
            bot.md.on_trade(SELL, D("4.0"), D("80000.0"), clock.t + i * 0.05)
        clock.t += 0.25

        # Regenerate ladder quotes during burst
        quotes = bot.engine.generate_ladder_quotes(bot._get_market(), bot.md, bot.ledger, clock.t, False, False)
        self.assertIsNotNone(quotes)
        print("✓ test_07_full_system_integration_with_hawkes_and_optimal_control passed: End-to-end integration verified.")


if __name__ == "__main__":
    unittest.main()
