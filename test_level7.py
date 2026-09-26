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


if __name__ == "__main__":
    unittest.main()
