"""Order Management and Execution Discipline.

Implements:
1. Fast Retreat: Instant cancellation on adverse price drift (>= retreat_drift_bps)
2. Lazy Advance: Deliberate repricing preserving queue priority (>= advance_drift_bps)
3. Anti-churn rate limiting and post-only safety
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from config import Config
from models import MarketSnapshot

log = logging.getLogger("VolatileMM.orders")


class OrderManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active_quotes: Dict[str, Optional[Tuple[str, float, float]]] = {"BUY": None, "SELL": None}
        self.last_requote_ts: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

    def sync_quotes(
        self,
        want_bid: Optional[float],
        want_ask: Optional[float],
        bid_qty: float,
        ask_qty: float,
        state: MarketSnapshot,
        adapter: Any,
    ) -> None:
        """Synchronizes target quotes with active exchange/simulator orders."""
        now = state.time
        open_orders = state.bot_open_orders
        existing_bid = None
        existing_ask = None
        for oid, o in open_orders.items():
            if o["side"] == "BUY": existing_bid = (oid, o["price"], o["qty"])
            elif o["side"] == "SELL": existing_ask = (oid, o["price"], o["qty"])

        # --- Manage BID Order --- #
        if want_bid is None:
            if existing_bid:
                adapter.cancel_order(existing_bid[0])
                self.active_quotes["BUY"] = None
        else:
            if existing_bid is None:
                oid = adapter.place_order("BUY", want_bid, bid_qty)
                self.active_quotes["BUY"] = (oid, want_bid, bid_qty)
                self.last_requote_ts["BUY"] = now
            else:
                drift_bps = abs(want_bid - existing_bid[1]) / state.mid * 10000.0
                is_retreating = (want_bid < existing_bid[1])  # Backing away from market
                is_advancing = (want_bid > existing_bid[1])   # Moving closer to market

                should_requote = False
                if is_retreating and drift_bps >= self.cfg.retreat_drift_bps:
                    should_requote = True  # Fast retreat to avoid adverse fill
                elif is_advancing and drift_bps >= self.cfg.advance_drift_bps and (now - self.last_requote_ts["BUY"] >= self.cfg.min_requote_interval_s):
                    should_requote = True  # Deliberate advance preserving queue

                if should_requote:
                    adapter.cancel_order(existing_bid[0])
                    oid = adapter.place_order("BUY", want_bid, bid_qty)
                    self.active_quotes["BUY"] = (oid, want_bid, bid_qty)
                    self.last_requote_ts["BUY"] = now

        # --- Manage ASK Order --- #
        if want_ask is None:
            if existing_ask:
                adapter.cancel_order(existing_ask[0])
                self.active_quotes["SELL"] = None
        else:
            if existing_ask is None:
                oid = adapter.place_order("SELL", want_ask, ask_qty)
                self.active_quotes["SELL"] = (oid, want_ask, ask_qty)
                self.last_requote_ts["SELL"] = now
            else:
                drift_bps = abs(want_ask - existing_ask[1]) / state.mid * 10000.0
                is_retreating = (want_ask > existing_ask[1])  # Backing away from market
                is_advancing = (want_ask < existing_ask[1])   # Moving closer to market

                should_requote = False
                if is_retreating and drift_bps >= self.cfg.retreat_drift_bps:
                    should_requote = True  # Fast retreat to avoid adverse fill
                elif is_advancing and drift_bps >= self.cfg.advance_drift_bps and (now - self.last_requote_ts["SELL"] >= self.cfg.min_requote_interval_s):
                    should_requote = True  # Deliberate advance preserving queue

                if should_requote:
                    adapter.cancel_order(existing_ask[0])
                    oid = adapter.place_order("SELL", want_ask, ask_qty)
                    self.active_quotes["SELL"] = (oid, want_ask, ask_qty)
                    self.last_requote_ts["SELL"] = now
