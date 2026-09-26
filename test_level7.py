"""Comprehensive Test Suite for Level 7 Market Maker Engine."""
import asyncio
import os
import sys
import unittest
from decimal import Decimal as D

import sim
from utils import BUY, SELL, fmt


class TestLevel7MarketMaker(unittest.IsolatedAsyncioTestCase):

    async def test_01_multi_ladder_placement(self):
        """Test that multiple quotes (ladder pairs) are maintained simultaneously and tracked individually."""
        bot, s, clock = sim.make(EXTRA_LEVELS=1, ORDER_USD=20, MAX_POSITION_USD=100, SKEW_BPS=0)
        await sim.step(bot, s, clock, "80000.0", "80100.0")

        buy_orders = bot.om.side_orders(BUY)
        sell_orders = bot.om.side_orders(SELL)

        self.assertGreaterEqual(len(buy_orders), 1, "Should have at least 1 buy order")
        self.assertGreaterEqual(len(sell_orders), 1, "Should have at least 1 sell order")
        
        for o in bot.om.orders.values():
            self.assertIn(o.pair_index, [0, 1])
            self.assertIn(o.side, [BUY, SELL])
            self.assertGreater(o.price, D(0))
            self.assertGreater(o.remaining, D(0))
        print("✓ test_01_multi_ladder_placement passed: Multiple ladder pairs placed and tracked individually.")

    async def test_02_orderbook_intelligence_microprice(self):
        """Test Level 5 Order-Book Intelligence: heavy buy pressure shifts fair value and protects the ask."""
        bot, s, clock = sim.make(EXTRA_LEVELS=0, ENABLE_ORDERBOOK_INTEL=1, USE_MICRO=1)
        
        await sim.step(bot, s, clock, "80000.0", "80100.0", bsz="1", asz="1")
        initial_ask = bot.om.get_order_by_slot(0, SELL)
        self.assertIsNotNone(initial_ask, "Initial ask should be present in balanced market")

        # Heavy bid pressure arrives: bid size = 10, ask size = 0.5 (microprice pumps toward ask)
        await sim.step(bot, s, clock, "80000.0", "80100.0", bsz="10", asz="0.5")
        
        new_ask = bot.om.get_order_by_slot(0, SELL)
        if new_ask is None:
            print("✓ test_02_orderbook_intelligence passed: Toxic ask pulled completely (EV < 0 protection).")
        else:
            self.assertGreater(new_ask.price, initial_ask.price, "Ask should reprice higher to protect against toxic buying")
            print(f"✓ test_02_orderbook_intelligence passed: Ask lifted from {initial_ask.price} to {new_ask.price}.")

    async def test_03_adaptive_ev_filter(self):
        """Test Level 4 Adaptive Market Making: quote only when EV > min_ev_bps."""
        bot, s, clock = sim.make(EXTRA_LEVELS=0, MIN_EV_BPS="1.0", ENABLE_ADAPTIVE_EV="1",
                                 MIN_EDGE_BPS="5", PENNY="0")
        
        await sim.step(bot, s, clock, "80000.0", "80000.1")
        self.assertEqual(s.rejects, 0, "No post-only crossing rejects should occur")
        print("✓ test_03_adaptive_ev_filter passed: Zero crossing rejects with adaptive EV guard.")

    async def test_04_online_learning_toxicity(self):
        """Test Level 6 Online Learning: markout tracking measures toxicity and widens edge."""
        bot, s, clock = sim.make(EXTRA_LEVELS=0, MARKOUT_HORIZON_S=1.0, TOX_MULT=2.0, ENABLE_ONLINE_LEARNING=1)
        await sim.step(bot, s, clock, "80000.0", "80050.0")

        fill = bot.ledger.on_fill(BUY, D("0.0003"), D("80000.0"), D("80025.0"), clock.t, D("5"))
        
        clock.t += 1.5
        bot.ledger.process_markouts(D("79900.0"), clock.t)
        
        self.assertGreater(bot.ledger.tox_bps, D(0), "Toxicity should be learned from adverse markout")
        side_tox = bot.ledger.side_tox_bps(BUY)
        self.assertGreater(side_tox, D(10), "Buy side toxicity should be elevated")
        print(f"✓ test_04_online_learning_toxicity passed: Learned buy-side toxicity = {side_tox:.2f} bps.")

    async def test_05_spread_capture_roundtrip(self):
        """Test that roundtrip fills capture positive spread without rejects."""
        bot, s, clock = sim.make(EXTRA_LEVELS=0, ORDER_USD=20, MAX_POSITION_USD=100)
        await sim.step(bot, s, clock, "80000.0", "80080.0")

        # Taker hits our bid
        s.taker(SELL)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertGreater(bot.ledger.position, D(0), "Should be long after bid fill")
        
        # Quoting exit at the ask
        await sim.step(bot, s, clock, "80000.0", "80080.0")
        s.taker(BUY)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        
        self.assertTrue(bot.ledger.is_flat(D("80040.0"), D("5.0")), f"Should be flat after closing ask fill, got {bot.ledger.position}")
        self.assertGreater(bot.ledger.realized, D(0), "Realized PnL from spread capture should be positive")
        self.assertEqual(s.rejects, 0, "Zero post-only rejects")
        print(f"✓ test_05_spread_capture_roundtrip passed: Realized PnL = ${bot.ledger.realized:.4f}.")




    async def test_06_toxic_regime_and_sweep_guard(self):
        """Test Level 7 Toxic Regime protection and preemptive Sweep Guard."""
        bot, s, clock = sim.make(EXTRA_LEVELS=2, ORDER_USD=20, MAX_POSITION_USD=200,
                                 REGIME_TOXIC_SPREAD_MULT="1.5", SWEEP_GUARD_FILLS=2)
        # 1. Normal step
        await sim.step(bot, s, clock, "80000.0", "80100.0")
        self.assertGreaterEqual(len(bot.om.side_orders(BUY)), 2)

        # 2. Simulate sweep fills (2 fills in <= 1.0s)
        f1 = bot.om.get_order_by_slot(0, BUY)
        bot._on_fill(BUY, f1.remaining, f1.price, f1)
        f2 = bot.om.get_order_by_slot(1, BUY)
        bot._on_fill(BUY, f2.remaining, f2.price, f2)
        await asyncio.sleep(0.01)
        self.assertGreater(bot._burst_blocked_until[BUY], clock.t, "BUY side should be sweep blocked")

        # 3. Test toxic regime OBI suppression
        bot.ledger.markouts.append(D("-5.0"))
        self.assertEqual(bot.md.detect_regime(clock.t, bot.ledger.tox_bps), "REGIME_D_TOXIC")
        # Unblock and test step under toxic sell dump (OBI < -0.4)
        bot._burst_blocked_until[BUY] = 0.0
        await sim.step(bot, s, clock, "80000.0", "80100.0", bsz="0.1", asz="10.0")
        buy_orders = bot.om.side_orders(BUY)
        self.assertEqual(len(buy_orders), 0, "BUY orders should be suppressed during toxic sell dump")
        print("✓ test_06_toxic_regime_and_sweep_guard passed: Toxic OBI protection and Sweep Guard active.")


    async def test_07_online_learning_full_adaptation_and_persistence(self):
        """Test Level 6+ Online Learning: adapt edge, spacing, sizing, skew, EV, and persist state."""
        test_path = 'test_learning_state_unit.json'
        if os.path.exists(test_path):
            os.remove(test_path)

        bot, s, clock = sim.make(EXTRA_LEVELS=2, ORDER_USD=20, MAX_POSITION_USD=200,
                                 ENABLE_ONLINE_LEARNING=1, MARKOUT_HORIZON_S=1.0,
                                 LEARNING_STATE_PATH=test_path)
        learner = bot.ledger.learner
        base_edge = learner.min_edge_bps
        base_spacing = learner.level_spacing_bps
        base_skew = learner.skew_bps

        # 1. Fill and adverse markout -> adapts edge, spacing, tox_mult, min_ev
        fill = bot.ledger.on_fill(BUY, D('0.0003'), D('80000.0'), D('80025.0'), clock.t, D('5'))
        clock.t += 2.0
        bot.ledger.process_markouts(D('79900.0'), clock.t) # -12.5 bps adverse markout

        self.assertGreater(learner.min_edge_bps, base_edge, "min_edge should widen on adverse markout")
        self.assertGreater(learner.level_spacing_bps, base_spacing, "ladder spacing should widen")
        self.assertGreater(learner.tox_mult, D('1.0'), "tox_mult should increase")
        self.assertGreater(learner.min_ev_bps, D('0.2'), "min_ev should increase")

        # 2. Inventory holding duration -> adapts skew_bps & gamma_risk_aversion
        learner.on_fill(BUY, D('80000.0'), D('80000.0'), D('150.0'), 60.0)
        self.assertGreater(learner.skew_bps, base_skew, "skew should adapt higher on prolonged inventory")

        # 3. Flow correlation -> adapts obi_alpha & tfi_beta
        base_obi = learner.obi_alpha
        learner.on_flow_correlation(D('0.5'), D('0.5'), D('1.0'))
        self.assertGreater(learner.obi_alpha, base_obi, "obi_alpha should adapt higher on predictive flow")

        # 4. Persistence verification
        self.assertTrue(os.path.exists(test_path), "Learning state file must be persisted to disk")

        # 5. Cold reload verification
        bot2, _, _ = sim.make(EXTRA_LEVELS=2, ORDER_USD=20, MAX_POSITION_USD=200,
                              ENABLE_ONLINE_LEARNING=1, LEARNING_STATE_PATH=test_path)
        self.assertEqual(bot2.ledger.learner.min_edge_bps, learner.min_edge_bps)
        self.assertEqual(bot2.ledger.learner.skew_bps, learner.skew_bps)
        self.assertEqual(bot2.ledger.learner.total_learned_updates, learner.total_learned_updates)

        if os.path.exists(test_path):
            os.remove(test_path)
        print("✓ test_07_online_learning_full_adaptation_and_persistence passed: All parameters adapted and persisted.")


    async def test_08_profitable_unwind_and_no_loss_selling(self):
        """Test that unwinding inventory guarantees minimum profit and does not sell at a loss."""
        bot, s, clock = sim.make(EXTRA_LEVELS=0, ORDER_USD=20, MAX_POSITION_USD=100,
                                 EXIT_MIN_PROFIT_BPS="1.5", STRESS_LOSS_BPS="20.0",
                                 MIN_REQUOTE_S="0.1", JUMP_BPS="20.0")
        await sim.step(bot, s, clock, "80000.0", "80080.0")
        s.taker(SELL) # fills long at 80000.1
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Market drops below entry price
        clock.t += 0.5
        await sim.step(bot, s, clock, "79980.0", "80000.0")
        ask_order = bot.om.get_order_by_slot(0, SELL)
        self.assertIsNotNone(ask_order)
        self.assertGreaterEqual(ask_order.price, D('80012.0'), "Bot must quote exit at or above min_profit_px")
        print("✓ test_08_profitable_unwind_and_no_loss_selling passed: Bot preserves profit and prevents loss selling.")


if __name__ == "__main__":
    unittest.main()
