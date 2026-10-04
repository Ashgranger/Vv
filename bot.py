"""Main Market Maker Bot Implementation."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any, Dict, Optional

from config import Config
from models import MarketSnapshot, BotFill
from pricing import PricingEngine
from risk import RiskGuard
from accounting import AccountingTracker
from orders import OrderManager

log = logging.getLogger("VolatileMM.bot")


class VolatileMarketMaker:
    def __init__(self, cfg: Config, adapter: Any):
        self.cfg = cfg
        self.adapter = adapter
        self.pricing = PricingEngine(cfg)
        self.risk = RiskGuard(cfg)
        self.accounting = AccountingTracker(cfg)
        self.om = OrderManager(cfg)

        self.stop_requested = False

    def on_tick(self, state: MarketSnapshot) -> None:
        """Processes a single event loop tick."""
        now = state.time
        self.risk.register_mid(now, state.mid)

        # 1. Circuit Breaker Evaluation
        if self.risk.check_circuit_breaker(state):
            self.adapter.cancel_all_orders()
            self.stop_requested = True
            return

        # 2. Synchronize Fills & Process Markouts
        fills = self.adapter.get_fills()
        if fills:
            last_fill = fills[-1]
            if not self.risk.recent_fills or self.risk.recent_fills[-1][0] != last_fill.timestamp:
                self.risk.on_fill(last_fill, now)

        self.accounting.process_markouts(state.mid, now)
        buy_tox_bps, sell_tox_bps = self.accounting.get_toxicity_bps()

        # 3. Risk Guards & Stress Unwind Check
        buy_blocked, sell_blocked, dir_pressure_bps = self.risk.evaluate_guards(state)
        is_stressed = self.risk.is_stressed(state, self.accounting.position_opened_ts)

        # 4. Generate Optimal Target Quotes
        want_bid, want_ask, bid_qty, ask_qty = self.pricing.compute_quotes(
            state=state,
            buy_blocked=buy_blocked,
            sell_blocked=sell_blocked,
            is_stressed=is_stressed,
            dir_pressure_bps=dir_pressure_bps,
            buy_tox_bps=buy_tox_bps,
            sell_tox_bps=sell_tox_bps,
        )

        # 5. Synchronize with Execution Adapter
        self.om.sync_quotes(want_bid, want_ask, bid_qty, ask_qty, state, self.adapter)

    def run_sim(self, duration_s: float, dt: float = 0.05) -> dict:
        """Runs the bot against the volatile market simulation environment."""
        n_steps = int(duration_s / dt)
        log.info(f"Starting simulation run ({duration_s:.0f}s, {n_steps} ticks)...")

        last_log_ts = 0.0
        for step in range(n_steps):
            snapshot = self.adapter.step(dt)
            self.on_tick(snapshot)
            if self.stop_requested:
                break

            if snapshot.time - last_log_ts >= 30.0:
                last_log_ts = snapshot.time
                log.info(
                    f"[T+{snapshot.time:>5.1f}s] Mid: ${snapshot.mid:.1f} | Spread: {snapshot.spread_bps:.2f} bps | "
                    f"Pos: {snapshot.bot_position:+.5f} BTC | PnL: ${snapshot.bot_total_pnl:+.2f}"
                )

        # Settle forward markouts
        for _ in range(120): self.adapter.step(dt)

        analytics = self.adapter.get_analytics()
        return analytics
