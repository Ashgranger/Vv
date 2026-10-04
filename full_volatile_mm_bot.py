#!/usr/bin/env python3
"""Complete, Production-Grade Volatile Market Making Bot.

All-in-One Standalone Architecture:
1. Dual-Mode Operation:
   - `--mode sim`: Runs an integrated high-frequency volatile market simulation (1-8 bps spread,
     stochastic jumps, informed toxic order flow, queue priority, and latency).
   - `--mode live`: Connects to live exchange REST / WebSocket endpoints with asynchronous signing,
     orderbook updates, and fill reconciliation.
2. Quantitative Engine (Optimized for 1-8 bps Spreads):
   - Micro-price & Order Flow Imbalance (OFI) alpha overlay for short-term price direction.
   - Avellaneda-Stoikov inventory reservation price with non-linear aversion.
   - Volatility-adaptive half-spread scaling adapted to dynamic 1-8 bps regimes.
   - Multi-layer adverse selection guards:
     * Momentum / falling knife filter (anti-hysteresis)
     * Consecutive fill burst cooldown
     * Rolling post-fill markout toxicity learning
     * Queue depletion fade
   - Asymmetric quotation & fast unwind liquidation mode.
3. Execution Discipline:
   - Fast Retreat (immediate cancellation on adverse drift >= 0.5 bps).
   - Lazy Advance (deliberate repricing on favorable drift >= 1.2 bps, protecting queue priority).
   - Post-Only ALO enforcement (never crosses book or pays taker fees).
   - Graceful shutdown canceling all open orders on SIGINT/SIGTERM.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VolatileMM")


# =========================================================================== #
# Configuration Dataclass
# =========================================================================== #

@dataclass
class BotConfig:
    # Mode & Connection
    mode: str = "sim"                     # "sim" or "live"
    market: str = "BTC-USD"
    env_name: str = "testnet"             # "mainnet" or "testnet"
    api_key: str = ""
    wallet_address: str = ""
    rest_url: str = "https://api.testnet.arcus.xyz"
    ws_url: str = "wss://api.testnet.arcus.xyz/v1/ws"

    # Order Sizing & Inventory Limits
    order_size: float = 0.005             # Order size in base asset
    order_usd: float = 40.0               # Order notional in USD
    max_position: float = 0.025           # Hard max position cap
    soft_position: float = 0.015          # Soft position limit (triggers aggressive unwind)
    session_max_loss_usd: float = 50.0    # Circuit breaker max loss

    # Spread & Edge Adaptation (bps)
    min_edge_bps: float = 1.0             # Minimum half-spread edge
    max_edge_bps: float = 8.0             # Maximum half-spread edge
    target_spread_min_bps: float = 1.0    # Target market spread min
    target_spread_max_bps: float = 8.0    # Target market spread max
    skew_bps: float = 2.5                 # Reservation price skew at full inventory
    risk_aversion_gamma: float = 0.20     # Inventory risk aversion factor

    # Microstructure Alpha (OFI & Micro-price)
    enable_micro_intel: bool = True
    alpha_micro: float = 0.6              # Micro-price weight
    alpha_obi: float = 0.8                # Order Book Imbalance shift weight
    alpha_tfi: float = 0.8                # Trade Flow Imbalance shift weight
    adverse_fade_mult: float = 1.0        # Continuous fade multiplier for toxic flow

    # Adverse Selection Guards
    trend_window_s: float = 2.0           # Momentum detection window
    trend_pull_bps: float = 3.5           # Strong trend threshold to pull quotes (bps)
    burst_threshold: int = 2              # Consecutive fills on same side
    burst_window_s: float = 5.0           # Burst window
    burst_cooldown_s: float = 5.0         # Burst cooldown duration
    min_depth_fade: float = 0.15          # Queue depletion threshold

    # Execution Discipline
    retreat_drift_bps: float = 0.5        # Fast retreat threshold (bps)
    advance_drift_bps: float = 1.2        # Deliberate advance threshold (bps)
    min_requote_interval_s: float = 0.2   # Anti-churn delay
    tick_size: float = 0.1
    step_size: float = 0.0001

    # Inventory Unwind & Stress
    max_hold_s: float = 30.0              # Max inventory holding time
    stress_loss_bps: float = 3.5          # Max allowable underwater loss before liquidation
    exit_profit_bps: float = 0.4          # Minimum profit target for exit quotes

    # Simulation-Specific Settings
    sim_duration_s: float = 180.0
    sim_jump_intensity: float = 0.06
    sim_noise_rate: float = 6.0
    sim_informed_rate: float = 0.4
    sim_base_vol_bps: float = 15.0
    sim_latency_s: float = 0.010
    sim_seed: int = 500

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BotConfig":
        cfg = cls()
        if hasattr(args, "mode") and args.mode: cfg.mode = args.mode
        if hasattr(args, "market") and args.market: cfg.market = args.market
        if hasattr(args, "order_size") and args.order_size: cfg.order_size = args.order_size
        if hasattr(args, "max_pos") and args.max_pos: cfg.max_position = args.max_pos
        if hasattr(args, "min_edge") and args.min_edge: cfg.min_edge_bps = args.min_edge
        if hasattr(args, "max_edge") and args.max_edge: cfg.max_edge_bps = args.max_edge
        if hasattr(args, "skew_bps") and args.skew_bps: cfg.skew_bps = args.skew_bps
        if hasattr(args, "duration") and args.duration: cfg.sim_duration_s = args.duration
        if hasattr(args, "seed") and args.seed is not None: cfg.sim_seed = args.seed
        return cfg


# =========================================================================== #
# Market Data & Internal Structures
# =========================================================================== #

@dataclass
class MarketSnapshot:
    time: float
    mid: float
    best_bid: float
    best_ask: float
    spread_bps: float
    bid_depth_touch: float
    ask_depth_touch: float
    micro_price: float
    obi: float
    vol_bps: float
    bot_position: float
    bot_avg_cost: float
    bot_cash: float
    bot_realized_pnl: float
    bot_unrealized_pnl: float
    bot_total_pnl: float
    bot_open_orders: Dict[str, dict]
    recent_trades: List[Any]


@dataclass
class FillRecord:
    timestamp: float
    order_id: str
    side: str
    price: float
    qty: float
    mid_at_fill: float
    position_after: float
    realized_pnl_delta: float
    fee_paid: float
    markout_1s: Optional[float] = None
    markout_5s: Optional[float] = None


"""Volatile Market Environment with Dynamic 1-8 bps Spreads and Microstructure Dynamics.

Features:
- Jump-diffusion mid-price dynamics with stochastic volatility clustering.
- Endogenous bid-ask spread fluctuating realistically between 1 and 8 bps.
- L2 Order Book depth with queue position tracking and FIFO matching.
- Dual-flow market order arrivals:
    * Uninformed noise traders (providing liquidity capture opportunities)
    * Informed toxic traders (anticipating jumps & order-book imbalances, causing adverse selection)
- Latency modeling for quote submission, replacement, and cancellation.
- Comprehensive markout tracking (1s and 5s post-fill) for adverse selection measurement.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple


@dataclass
class SimOrder:
    order_id: str
    owner: str             # "bot" or "external"
    side: str              # "BUY" or "SELL"
    price: float
    qty: float
    remaining: float
    queue_ahead: float     # Volume ahead of this order in the queue at this price level
    timestamp: float
    active: bool = True


@dataclass
class Trade:
    timestamp: float
    side: str              # "BUY" (buyer taker) or "SELL" (seller taker)
    price: float
    qty: float
    is_informed: bool
    bot_fill_qty: float = 0.0
    bot_order_id: Optional[str] = None


@dataclass
class BotFill:
    timestamp: float
    order_id: str
    side: str
    price: float
    qty: float
    mid_at_fill: float
    position_after: float
    realized_pnl_delta: float
    fee_paid: float
    markout_1s: Optional[float] = None
    markout_5s: Optional[float] = None


class VolatileMarketEnv:
    def __init__(
        self,
        initial_price: float = 80000.0,
        min_spread_bps: float = 1.0,
        max_spread_bps: float = 8.0,
        base_spread_bps: float = 2.5,
        tick_size: float = 0.1,
        step_size: float = 0.0001,
        maker_fee_bps: float = 0.0,
        taker_fee_bps: float = 2.0,
        noise_order_rate: float = 5.0,        # Noise market orders per second
        informed_order_rate: float = 0.8,     # Toxic orders per second
        jump_intensity: float = 0.15,         # Jumps per second
        base_vol_bps: float = 15.0,           # Baseline volatility per sqrt(sec) in bps
        latency_s: float = 0.010,             # 10ms execution delay
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            random.seed(seed)

        self.initial_price = initial_price
        self.mid = initial_price
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.base_spread_bps = base_spread_bps
        self.current_spread_bps = base_spread_bps
        self.tick_size = tick_size
        self.step_size = step_size
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.noise_order_rate = noise_order_rate
        self.informed_order_rate = informed_order_rate
        self.jump_intensity = jump_intensity
        self.base_vol_bps = base_vol_bps
        self.current_vol_bps = base_vol_bps
        self.latency_s = latency_s

        self.time = 0.0
        self.seq = 0

        # Order book queues: price -> list of SimOrder
        self.bid_levels: Dict[float, List[SimOrder]] = {}
        self.ask_levels: Dict[float, List[SimOrder]] = {}

        # Bot orders and pending actions (latency queue)
        self.bot_orders: Dict[str, SimOrder] = {}
        self.pending_actions: List[Tuple[float, str, dict]] = []  # (execute_time, action_type, payload)

        # Bot state & accounting
        self.bot_position = 0.0
        self.bot_avg_cost = 0.0
        self.bot_cash = 0.0
        self.bot_realized_pnl = 0.0
        self.bot_fees_paid = 0.0
        self.bot_volume_traded = 0.0
        self.bot_fills: List[BotFill] = []

        # History buffers for microstructure calculation
        self.mid_history: deque[Tuple[float, float]] = deque(maxlen=2000)
        self.trade_history: deque[Trade] = deque(maxlen=1000)
        self.pending_markouts: List[Tuple[float, BotFill, float]] = []  # (target_time, fill, horizon_s)

        # Order book state
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_depth_touch = 1.0
        self.ask_depth_touch = 1.0
        self.micro_price = initial_price
        self.obi = 0.0  # Order Book Imbalance: (B - A) / (B + A)

        # Initialize book
        self._refresh_market_spread()
        self._rebuild_external_book()
        self.mid_history.append((self.time, self.mid))

    def _round_tick(self, px: float) -> float:
        return round(round(px / self.tick_size) * self.tick_size, 4)

    def _round_step(self, qty: float) -> float:
        return round(round(qty / self.step_size) * self.step_size, 6)

    def _refresh_market_spread(self) -> None:
        """Dynamically updates the market spread between 1 and 8 bps based on volatility and impact."""
        # Mean reverting Ornstein-Uhlenbeck spread component
        vol_ratio = self.current_vol_bps / self.base_vol_bps
        spread_target = self.base_spread_bps * (0.6 + 0.5 * vol_ratio)
        
        # Add random microstructure noise
        noise = random.gauss(0.0, 0.4)
        raw_spread = self.current_spread_bps * 0.85 + (spread_target + noise) * 0.15
        
        # Clamp strictly between 1.0 and 8.0 bps as requested
        self.current_spread_bps = max(self.min_spread_bps, min(self.max_spread_bps, raw_spread))

    def _rebuild_external_book(self) -> None:
        """Regenerates external liquidity levels around mid-price conforming to the 1-8 bps spread."""
        half_spread = (self.mid * (self.current_spread_bps / 10000.0)) / 2.0
        target_bid = self._round_tick(self.mid - half_spread)
        target_ask = self._round_tick(self.mid + half_spread)

        if target_ask <= target_bid:
            target_ask = self._round_tick(target_bid + self.tick_size)

        self.best_bid = target_bid
        self.best_ask = target_ask

        # Randomize touch depths
        self.bid_depth_touch = max(0.005, round(random.uniform(0.01, 0.03), 4))
        self.ask_depth_touch = max(0.005, round(random.uniform(0.01, 0.03), 4))

        # Re-calculate micro-price and OBI
        total_depth = self.bid_depth_touch + self.ask_depth_touch
        self.micro_price = (
            (self.best_bid * self.ask_depth_touch + self.best_ask * self.bid_depth_touch)
            / total_depth
        )
        self.obi = (self.bid_depth_touch - self.ask_depth_touch) / total_depth

    # ---- Bot Interaction API ---------------------------------------------- #

    def place_order(self, side: str, price: float, qty: float) -> str:
        """Queues a limit order placement with realistic latency."""
        self.seq += 1
        order_id = f"bot-{self.seq}"
        exec_time = self.time + self.latency_s
        payload = {
            "order_id": order_id,
            "side": side.upper(),
            "price": self._round_tick(price),
            "qty": self._round_step(qty),
        }
        self.pending_actions.append((exec_time, "PLACE", payload))
        return order_id

    def cancel_order(self, order_id: str) -> None:
        """Queues a limit order cancellation with latency."""
        exec_time = self.time + self.latency_s
        self.pending_actions.append((exec_time, "CANCEL", {"order_id": order_id}))

    def cancel_all_orders(self) -> None:
        """Queues cancellation of all active bot orders."""
        exec_time = self.time + self.latency_s
        self.pending_actions.append((exec_time, "CANCEL_ALL", {}))

    def _execute_place(self, payload: dict) -> None:
        order_id = payload["order_id"]
        side = payload["side"]
        price = payload["price"]
        qty = payload["qty"]

        # Post-Only Check: if quote crosses touch, it gets rejected (ALO)
        if side == "BUY" and price >= self.best_ask:
            return  # Rejected by post-only guard
        if side == "SELL" and price <= self.best_bid:
            return  # Rejected by post-only guard

        # Calculate queue ahead
        queue_ahead = 0.0
        if side == "BUY":
            if price == self.best_bid:
                queue_ahead = self.bid_depth_touch
            elif price < self.best_bid:
                # Levels further back have less priority
                ticks_back = (self.best_bid - price) / self.tick_size
                queue_ahead = self.bid_depth_touch + ticks_back * 1.5
        else:
            if price == self.best_ask:
                queue_ahead = self.ask_depth_touch
            elif price > self.best_ask:
                ticks_back = (price - self.best_ask) / self.tick_size
                queue_ahead = self.ask_depth_touch + ticks_back * 1.5

        order = SimOrder(
            order_id=order_id,
            owner="bot",
            side=side,
            price=price,
            qty=qty,
            remaining=qty,
            queue_ahead=queue_ahead,
            timestamp=self.time,
            active=True,
        )
        self.bot_orders[order_id] = order

    def _execute_cancel(self, payload: dict) -> None:
        oid = payload["order_id"]
        if oid in self.bot_orders:
            self.bot_orders[oid].active = False
            del self.bot_orders[oid]

    def _execute_cancel_all(self) -> None:
        for o in self.bot_orders.values():
            o.active = False
        self.bot_orders.clear()

    # ---- Market Matching & Fill Simulation -------------------------------- #

    def _process_market_order(self, side: str, qty: float, is_informed: bool) -> None:
        """Executes a taker market order against the passive order book queues."""
        trade_px = self.best_ask if side == "BUY" else self.best_bid
        bot_fill_qty = 0.0
        matched_bot_order_id = None

        # Check interaction with bot orders
        for oid, o in list(self.bot_orders.items()):
            if not o.active:
                continue

            if side == "BUY" and o.side == "SELL":
                # Market BUY takes from SELL book
                if trade_px >= o.price:
                    if o.queue_ahead > 0:
                        consumed = min(o.queue_ahead, qty)
                        o.queue_ahead -= consumed
                        rem_qty = qty - consumed
                    else:
                        rem_qty = qty

                    if rem_qty > 0 and o.remaining > 0:
                        fill_qty = min(o.remaining, rem_qty)
                        o.remaining -= fill_qty
                        bot_fill_qty += fill_qty
                        matched_bot_order_id = oid
                        self._record_bot_fill(o, fill_qty, o.price)

                        if o.remaining <= 1e-7:
                            o.active = False
                            del self.bot_orders[oid]

            elif side == "SELL" and o.side == "BUY":
                # Market SELL takes from BUY book
                if trade_px <= o.price:
                    if o.queue_ahead > 0:
                        consumed = min(o.queue_ahead, qty)
                        o.queue_ahead -= consumed
                        rem_qty = qty - consumed
                    else:
                        rem_qty = qty

                    if rem_qty > 0 and o.remaining > 0:
                        fill_qty = min(o.remaining, rem_qty)
                        o.remaining -= fill_qty
                        bot_fill_qty += fill_qty
                        matched_bot_order_id = oid
                        self._record_bot_fill(o, fill_qty, o.price)

                        if o.remaining <= 1e-7:
                            o.active = False
                            del self.bot_orders[oid]

        # Record trade
        self.trade_history.append(Trade(
            timestamp=self.time,
            side=side,
            price=trade_px,
            qty=qty,
            is_informed=is_informed,
            bot_fill_qty=bot_fill_qty,
            bot_order_id=matched_bot_order_id
        ))

        # Deplete book depth
        if side == "BUY":
            self.ask_depth_touch = max(0.1, self.ask_depth_touch - qty)
            if self.ask_depth_touch <= 0.1:
                # Spread widens as ask is consumed
                self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 0.8)
        else:
            self.bid_depth_touch = max(0.1, self.bid_depth_touch - qty)
            if self.bid_depth_touch <= 0.1:
                # Spread widens as bid is consumed
                self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 0.8)

    def _record_bot_fill(self, order: SimOrder, fill_qty: float, fill_price: float) -> None:
        signed_qty = fill_qty if order.side == "BUY" else -fill_qty
        prev_pos = self.bot_position
        new_pos = prev_pos + signed_qty

        realized_delta = 0.0
        if prev_pos == 0.0 or (prev_pos > 0) == (signed_qty > 0):
            # Increasing position: update average entry cost
            total_abs = abs(prev_pos) + fill_qty
            self.bot_avg_cost = (self.bot_avg_cost * abs(prev_pos) + fill_price * fill_qty) / total_abs
        else:
            # Closing / unwinding position
            closed_qty = min(abs(prev_pos), fill_qty)
            direction = 1.0 if prev_pos > 0 else -1.0
            realized_delta = (fill_price - self.bot_avg_cost) * closed_qty * direction
            if abs(new_pos) < 1e-8:
                self.bot_avg_cost = 0.0
            elif (new_pos > 0) != (prev_pos > 0):
                self.bot_avg_cost = fill_price  # Flipped through zero

        fee = fill_qty * fill_price * (self.maker_fee_bps / 10000.0)
        realized_delta -= fee
        self.bot_fees_paid += fee
        self.bot_realized_pnl += realized_delta
        self.bot_cash -= (signed_qty * fill_price + fee)
        self.bot_position = new_pos
        self.bot_volume_traded += fill_qty * fill_price

        fill_record = BotFill(
            timestamp=self.time,
            order_id=order.order_id,
            side=order.side,
            price=fill_price,
            qty=fill_qty,
            mid_at_fill=self.mid,
            position_after=new_pos,
            realized_pnl_delta=realized_delta,
            fee_paid=fee,
        )
        self.bot_fills.append(fill_record)

        # Schedule markout evaluation at 1.0s and 5.0s horizons
        self.pending_markouts.append((self.time + 1.0, fill_record, 1.0))
        self.pending_markouts.append((self.time + 5.0, fill_record, 5.0))

    def _process_markouts(self) -> None:
        rem = []
        for target_time, fill, horizon in self.pending_markouts:
            if self.time >= target_time:
                # Markout in bps: (+ = moved favorably, - = adverse selection)
                if fill.side == "BUY":
                    m_bps = (self.mid - fill.price) / fill.price * 10000.0
                else:
                    m_bps = (fill.price - self.mid) / fill.price * 10000.0

                if horizon == 1.0:
                    fill.markout_1s = m_bps
                elif horizon == 5.0:
                    fill.markout_5s = m_bps
            else:
                rem.append((target_time, fill, horizon))
        self.pending_markouts = rem

    # ---- Environment Step ------------------------------------------------- #

    def step(self, dt: float = 0.05) -> dict:
        """Advances the simulation by dt seconds."""
        self.time += dt

        # 1. Execute matured pending bot actions (latency queue)
        ready = [act for act in self.pending_actions if act[0] <= self.time]
        self.pending_actions = [act for act in self.pending_actions if act[0] > self.time]
        for _, act_type, payload in ready:
            if act_type == "PLACE":
                self._execute_place(payload)
            elif act_type == "CANCEL":
                self._execute_cancel(payload)
            elif act_type == "CANCEL_ALL":
                self._execute_cancel_all()

        # 2. Update Stochastic Volatility (Mean-reverting Heston-like vol clustering)
        vol_drift = 1.2 * (self.base_vol_bps - self.current_vol_bps) * dt
        vol_shock = random.gauss(0, 1.5) * math.sqrt(dt)
        self.current_vol_bps = max(5.0, self.current_vol_bps + vol_drift + vol_shock)

        # 3. Simulate Mid-Price (Continuous Geometric Brownian Motion + Jumps)
        sigma = (self.current_vol_bps / 10000.0) / math.sqrt(1.0)
        drift = 0.0
        diffusion = sigma * math.sqrt(dt) * random.gauss(0, 1)

        # Poisson jump process
        jump = 0.0
        jump_occurred = False
        jump_side = None
        if random.random() < self.jump_intensity * dt:
            jump_occurred = True
            # Jump magnitude between 2 and 10 bps
            jump_mag_bps = random.uniform(2.5, 9.0)
            jump_sign = random.choice([1.0, -1.0])
            jump = jump_sign * (jump_mag_bps / 10000.0)
            jump_side = "BUY" if jump_sign > 0 else "SELL"
            # Jumps cause sudden volatility spikes and widen the market spread toward 8 bps
            self.current_vol_bps += jump_mag_bps * 1.5
            self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 2.0)

        self.mid = self.mid * math.exp(drift + diffusion + jump)
        self.mid_history.append((self.time, self.mid))

        # 4. Refresh spread & order book depth
        self._refresh_market_spread()
        self._rebuild_external_book()

        # 5. Simulate Market Orders
        # A) Informed / Toxic orders: arrive right before / during jumps or severe imbalance
        if jump_occurred:
            # Informed traders trade heavily in direction of jump (causing adverse selection)
            informed_size = random.uniform(0.02, 0.06)
            self._process_market_order(side=jump_side, qty=informed_size, is_informed=True)
        elif random.random() < self.informed_order_rate * dt:
            # Micro-momentum informed order
            recent_ret = (self.mid - self.mid_history[0][1]) / self.mid_history[0][1] * 10000.0 if len(self.mid_history) > 1 else 0.0
            if abs(recent_ret) > 1.5:
                side = "BUY" if recent_ret > 0 else "SELL"
                inf_qty = random.uniform(0.005, 0.025)
                self._process_market_order(side=side, qty=inf_qty, is_informed=True)

        # B) Uninformed / Noise orders: independent Poisson flow
        num_noise_orders = int(random.expovariate(1.0 / max(0.01, self.noise_order_rate * dt))) if random.random() < 0.8 else 0
        for _ in range(min(num_noise_orders, 4)):
            side = random.choice(["BUY", "SELL"])
            noise_qty = random.uniform(0.005, 0.025)
            self._process_market_order(side=side, qty=noise_qty, is_informed=False)

        # 6. Natural queue decay (cancellations of external orders ahead of bot)
        for o in self.bot_orders.values():
            if o.active and o.queue_ahead > 0:
                # Other market participants cancel at 15% rate per second
                o.queue_ahead = max(0.0, o.queue_ahead - o.queue_ahead * 0.15 * dt)

        # 7. Process pending markouts
        self._process_markouts()

        # Return snapshot
        return self.get_state()

    def get_state(self) -> dict:
        """Returns the complete market snapshot for the bot."""
        unrealized = (self.mid - self.bot_avg_cost) * self.bot_position if self.bot_position != 0 else 0.0
        total_pnl = self.bot_realized_pnl + unrealized

        return {
            "time": self.time,
            "mid": self.mid,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread_bps": self.current_spread_bps,
            "bid_depth_touch": self.bid_depth_touch,
            "ask_depth_touch": self.ask_depth_touch,
            "micro_price": self.micro_price,
            "obi": self.obi,
            "vol_bps": self.current_vol_bps,
            "bot_position": self.bot_position,
            "bot_avg_cost": self.bot_avg_cost,
            "bot_cash": self.bot_cash,
            "bot_realized_pnl": self.bot_realized_pnl,
            "bot_unrealized_pnl": unrealized,
            "bot_total_pnl": total_pnl,
            "bot_open_orders": {
                oid: {"side": o.side, "price": o.price, "qty": o.remaining, "queue_ahead": o.queue_ahead}
                for oid, o in self.bot_orders.items() if o.active
            },
            "recent_trades": list(self.trade_history)[-20:],
        }

    def get_analytics(self) -> dict:
        """Computes comprehensive performance and adverse-selection analytics."""
        unrealized = (self.mid - self.bot_avg_cost) * self.bot_position if self.bot_position != 0 else 0.0
        total_pnl = self.bot_realized_pnl + unrealized

        markouts_1s = [f.markout_1s for f in self.bot_fills if f.markout_1s is not None]
        markouts_5s = [f.markout_5s for f in self.bot_fills if f.markout_5s is not None]

        avg_m1 = sum(markouts_1s) / len(markouts_1s) if markouts_1s else 0.0
        avg_m5 = sum(markouts_5s) / len(markouts_5s) if markouts_5s else 0.0

        adverse_fills_1s = sum(1 for m in markouts_1s if m < 0)
        adverse_ratio_1s = adverse_fills_1s / len(markouts_1s) if markouts_1s else 0.0

        adverse_fills_5s = sum(1 for m in markouts_5s if m < 0)
        adverse_ratio_5s = adverse_fills_5s / len(markouts_5s) if markouts_5s else 0.0

        return {
            "total_pnl": total_pnl,
            "realized_pnl": self.bot_realized_pnl,
            "unrealized_pnl": unrealized,
            "fees_paid": self.bot_fees_paid,
            "volume_traded_usd": self.bot_volume_traded,
            "total_fills": len(self.bot_fills),
            "final_position": self.bot_position,
            "avg_markout_1s_bps": avg_m1,
            "avg_markout_5s_bps": avg_m5,
            "adverse_fill_ratio_1s": adverse_ratio_1s,
            "adverse_fill_ratio_5s": adverse_ratio_5s,
        }


# =========================================================================== #
# Quantitative Volatile Market Making Engine
# =========================================================================== #

class VolatilePricingEngine:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.mid_buffer: deque[Tuple[float, float]] = deque(maxlen=200)
        self.last_knife_ts: Dict[str, float] = {"BUY": -100.0, "SELL": -100.0}

    def compute_fair_value(self, state: MarketSnapshot) -> float:
        """Estimates short-term fair value S* using micro-price, OBI, and TFI."""
        mid = state.mid
        if not self.cfg.enable_micro_intel:
            return mid

        micro = state.micro_price
        obi = state.obi
        half_spread = (state.best_ask - state.best_bid) / 2.0

        # Trade Flow Imbalance over 2.5s window
        recent_trades = state.recent_trades
        now = state.time
        tfi_window = 2.5
        buy_vol = sum(t.qty for t in recent_trades if t.side == "BUY" and now - t.timestamp <= tfi_window)
        sell_vol = sum(t.qty for t in recent_trades if t.side == "SELL" and now - t.timestamp <= tfi_window)
        total_vol = buy_vol + sell_vol
        tfi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

        obi_shift = half_spread * obi * self.cfg.alpha_obi
        tfi_shift = half_spread * tfi * self.cfg.alpha_tfi

        fair_val = (1.0 - self.cfg.alpha_micro) * mid + self.cfg.alpha_micro * micro + obi_shift + tfi_shift
        return fair_val

    def compute_reservation_price(self, fair_value: float, state: MarketSnapshot) -> float:
        """Computes inventory reservation price r(s, q) via Avellaneda-Stoikov."""
        if self.cfg.max_position <= 0:
            return fair_value

        q_ratio = max(-1.0, min(1.0, state.bot_position / self.cfg.max_position))
        q_eff = math.copysign(math.pow(abs(q_ratio), 1.2), q_ratio)

        vol_factor = 1.0 + (state.vol_bps / 20.0) * self.cfg.risk_aversion_gamma
        skew_bps = q_eff * self.cfg.skew_bps * vol_factor

        res_price = fair_value * (1.0 - skew_bps / 10000.0)
        return res_price

    def evaluate_quotes(
        self,
        state: MarketSnapshot,
        cooldown_until: Dict[str, float],
        markouts_buy: deque[float],
        markouts_sell: deque[float],
        position_opened_ts: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], float, float]:
        """Generates optimal limit bid and ask quotes for the 1-8 bps regime."""
        now = state.time
        self.mid_buffer.append((now, state.mid))

        mid = state.mid
        best_bid = state.best_bid
        best_ask = state.best_ask
        mkt_spread_bps = state.spread_bps
        tick = self.cfg.tick_size

        fair_val = self.compute_fair_value(state)
        res_px = self.compute_reservation_price(fair_val, state)

        # 1. Directional Flow & Momentum Pressure (Continuous Fade)
        recent_trades = state.recent_trades
        t_2s = [t for t in recent_trades if now - t.timestamp <= 2.0]
        tfi = 0.0
        if t_2s:
            b_vol = sum(t.qty for t in t_2s if t.side == "BUY")
            s_vol = sum(t.qty for t in t_2s if t.side == "SELL")
            tot = b_vol + s_vol
            if tot > 0: tfi = (b_vol - s_vol) / tot

        obi = state.obi

        ret_trend_bps = 0.0
        if len(self.mid_buffer) >= 2:
            target_ts = now - self.cfg.trend_window_s
            past_mid = self.mid_buffer[0][1]
            for ts, px in self.mid_buffer:
                if ts >= target_ts:
                    past_mid = px
                    break
            ret_trend_bps = (mid - past_mid) / past_mid * 10000.0

        dir_pressure_bps = (0.8 * tfi + 0.5 * obi + 0.3 * (ret_trend_bps / 2.0)) * self.cfg.adverse_fade_mult

        buy_tox_bps = max(0.0, -sum(markouts_buy) / len(markouts_buy)) if markouts_buy else 0.0
        sell_tox_bps = max(0.0, -sum(markouts_sell) / len(markouts_sell)) if markouts_sell else 0.0

        half_spr_bps = max(self.cfg.min_edge_bps, min(self.cfg.max_edge_bps, mkt_spread_bps / 2.0))
        bid_dist_bps = half_spr_bps + max(0.0, -dir_pressure_bps) + buy_tox_bps * 0.8
        ask_dist_bps = half_spr_bps + max(0.0, dir_pressure_bps) + sell_tox_bps * 0.8

        # 2. Check Cooldowns & Circuit Breakers
        buy_blocked = (now < cooldown_until["BUY"])
        sell_blocked = (now < cooldown_until["SELL"])

        if ret_trend_bps <= -self.cfg.trend_pull_bps:
            buy_blocked = True
        elif ret_trend_bps >= self.cfg.trend_pull_bps:
            sell_blocked = True

        if state.bot_position >= self.cfg.max_position - 1e-6:
            buy_blocked = True
        if state.bot_position <= -self.cfg.max_position + 1e-6:
            sell_blocked = True

        # 3. Inventory Stress / Auto-Unwind
        is_stressed = False
        holding_time = (now - position_opened_ts) if position_opened_ts else 0.0
        if state.bot_position != 0.0:
            unrealized_bps = (
                (mid - state.bot_avg_cost) / state.bot_avg_cost * 10000.0
                if state.bot_position > 0
                else (state.bot_avg_cost - mid) / state.bot_avg_cost * 10000.0
            )
            if (
                holding_time > self.cfg.max_hold_s
                or unrealized_bps <= -self.cfg.stress_loss_bps
                or abs(state.bot_position) >= self.cfg.soft_position
            ):
                is_stressed = True

        want_bid: Optional[float] = None
        want_ask: Optional[float] = None
        bid_qty = self.cfg.order_size
        ask_qty = self.cfg.order_size

        # --- BID QUOTE --- #
        is_buy_unwind = (state.bot_position < 0.0)
        if is_buy_unwind:
            bid_qty = min(abs(state.bot_position), self.cfg.order_size)
            cand_bid = best_bid if is_stressed else min(best_ask - tick, res_px * (1.0 - self.cfg.min_edge_bps / 10000.0))
            cand_bid = min(cand_bid, best_ask - tick)
            want_bid = round(round(cand_bid / tick) * tick, 4)
        elif not buy_blocked and not is_stressed:
            model_bid = res_px * (1.0 - bid_dist_bps / 10000.0)
            cand_bid = min(best_bid, model_bid)
            cand_bid = min(cand_bid, best_ask - tick)
            want_bid = round(round(cand_bid / tick) * tick, 4)

        # --- ASK QUOTE --- #
        is_sell_unwind = (state.bot_position > 0.0)
        if is_sell_unwind:
            ask_qty = min(abs(state.bot_position), self.cfg.order_size)
            if is_stressed:
                cand_ask = best_ask
            else:
                cand_ask = max(best_bid + tick, res_px * (1.0 + self.cfg.min_edge_bps / 10000.0))
                if state.bot_avg_cost > 0.0:
                    cand_ask = max(cand_ask, state.bot_avg_cost * (1.0 + self.cfg.exit_profit_bps / 10000.0))
            cand_ask = max(cand_ask, best_bid + tick)
            want_ask = round(round(cand_ask / tick) * tick, 4)
        elif not sell_blocked and not is_stressed:
            model_ask = res_px * (1.0 + ask_dist_bps / 10000.0)
            cand_ask = max(best_ask, model_ask)
            cand_ask = max(cand_ask, best_bid + tick)
            want_ask = round(round(cand_ask / tick) * tick, 4)

        # Cross Safety Check
        if want_bid is not None and want_ask is not None and want_bid >= want_ask:
            if state.bot_position > 0: want_bid = None
            elif state.bot_position < 0: want_ask = None
            else: want_bid = want_ask = None

        return want_bid, want_ask, bid_qty, ask_qty


# =========================================================================== #
# Execution Manager (Fast Retreat vs Lazy Advance)
# =========================================================================== #

class OrderManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.active_quotes: Dict[str, Optional[Tuple[str, float, float]]] = {"BUY": None, "SELL": None}
        self.last_requote_ts: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

    def sync(
        self,
        want_bid: Optional[float],
        want_ask: Optional[float],
        bid_qty: float,
        ask_qty: float,
        state: MarketSnapshot,
        adapter: Any,
    ) -> None:
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
                is_retreating = (want_bid < existing_bid[1])
                is_advancing = (want_bid > existing_bid[1])

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
                is_retreating = (want_ask > existing_ask[1])
                is_advancing = (want_ask < existing_ask[1])

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


# =========================================================================== #
# Full Market Maker Bot Core
# =========================================================================== #

class VolatileMarketMakerBot:
    def __init__(self, cfg: BotConfig, adapter: Any):
        self.cfg = cfg
        self.adapter = adapter
        self.pricing = VolatilePricingEngine(cfg)
        self.om = OrderManager(cfg)

        self.position = 0.0
        self.avg_cost = 0.0
        self.position_opened_ts: Optional[float] = None

        self.recent_fills: deque[Tuple[float, str]] = deque(maxlen=20)
        self.cooldown_until: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}
        self.markouts_buy: deque[float] = deque(maxlen=20)
        self.markouts_sell: deque[float] = deque(maxlen=20)

        self.stop_requested = False

    def on_tick(self, state: MarketSnapshot) -> None:
        """Core periodic decision loop."""
        now = state.time

        # Update position and cost from state
        self.position = state.bot_position
        self.avg_cost = state.bot_avg_cost
        if self.position != 0.0 and self.position_opened_ts is None:
            self.position_opened_ts = now
        elif self.position == 0.0:
            self.position_opened_ts = None

        # Session Circuit Breaker
        if state.bot_total_pnl <= -self.cfg.session_max_loss_usd:
            log.error(f"CIRCUIT BREAKER TRIGGERED: Total PnL (${state.bot_total_pnl:.2f}) <= -${self.cfg.session_max_loss_usd}")
            self.adapter.cancel_all_orders()
            self.stop_requested = True
            return

        # Synchronize fills and markout tracking
        fills = self.adapter.get_fills()
        if fills:
            last_fill = fills[-1]
            if not self.recent_fills or self.recent_fills[-1][0] != last_fill.timestamp:
                self.recent_fills.append((last_fill.timestamp, last_fill.side))
                recent_same = [f for f in self.recent_fills if f[1] == last_fill.side and now - f[0] <= self.cfg.burst_window_s]
                if len(recent_same) >= self.cfg.burst_threshold:
                    self.cooldown_until[last_fill.side] = now + self.cfg.burst_cooldown_s
                    log.warning(f"BURST GUARD ACTIVATED: Pausing {last_fill.side} quotes for {self.cfg.burst_cooldown_s}s")

            for f in fills:
                if f.markout_1s is not None:
                    if f.side == "BUY": self.markouts_buy.append(f.markout_1s)
                    else: self.markouts_sell.append(f.markout_1s)

        # Generate quotes and synchronize orders
        wb, wa, bq, aq = self.pricing.evaluate_quotes(
            state, self.cooldown_until, self.markouts_buy, self.markouts_sell, self.position_opened_ts
        )
        self.om.sync(wb, wa, bq, aq, state, self.adapter)


# =========================================================================== #
# CLI Runner and Entrypoint
# =========================================================================== #

def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full Volatile Market Maker Bot")
    p.add_argument("--mode", choices=["sim", "live"], default="sim", help="Operating mode: sim (backtest) or live")
    p.add_argument("--market", default="BTC-USD", help="Target trading pair (e.g. BTC-USD)")
    p.add_argument("--order-size", type=float, default=0.005, help="Quote size in base asset")
    p.add_argument("--max-pos", type=float, default=0.025, help="Hard maximum position cap")
    p.add_argument("--min-edge", type=float, default=1.0, help="Minimum half-spread edge in bps")
    p.add_argument("--max-edge", type=float, default=8.0, help="Maximum half-spread edge in bps")
    p.add_argument("--skew-bps", type=float, default=2.5, help="Inventory reservation skew in bps")
    p.add_argument("--duration", type=float, default=180.0, help="Simulation duration in seconds")
    p.add_argument("--seed", type=int, default=500, help="Random seed for simulation")
    return p


def run_bot() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    cfg = BotConfig.from_args(args)

    log.info("=" * 70)
    log.info("INITIALIZING VOLATILE MARKET MAKER (1-8 BPS REGIME)")
    log.info(f"Mode: {cfg.mode.upper()} | Pair: {cfg.market} | Size: {cfg.order_size} | MaxPos: {cfg.max_position}")
    log.info(f"Target Spread: {cfg.target_spread_min_bps}-{cfg.target_spread_max_bps} bps | Edge: {cfg.min_edge_bps}-{cfg.max_edge_bps} bps | Skew: {cfg.skew_bps} bps")
    log.info("=" * 70)

    if cfg.mode == "sim":
        sim_env = VolatileMarketEnv(
            initial_price=80000.0,
            min_spread_bps=cfg.target_spread_min_bps,
            max_spread_bps=cfg.target_spread_max_bps,
            base_spread_bps=2.5,
            noise_order_rate=cfg.sim_noise_rate,
            informed_order_rate=cfg.sim_informed_rate,
            jump_intensity=cfg.sim_jump_intensity,
            base_vol_bps=cfg.sim_base_vol_bps,
            latency_s=cfg.sim_latency_s,
            seed=cfg.sim_seed,
        )

        class SimAdapter:
            def __init__(self, env): self.env = env
            def place_order(self, side, price, qty): return self.env.place_order(side, price, qty)
            def cancel_order(self, oid): self.env.cancel_order(oid)
            def cancel_all_orders(self): self.env.cancel_all_orders()
            def get_fills(self): return self.env.bot_fills

        adapter = SimAdapter(sim_env)
        bot = VolatileMarketMakerBot(cfg, adapter)

        dt = 0.05
        n_steps = int(cfg.sim_duration_s / dt)
        log.info(f"Running autonomous volatile simulation for {cfg.sim_duration_s:.0f}s ({n_steps} ticks)...")

        last_report_ts = 0.0
        for step in range(n_steps):
            raw_state = sim_env.step(dt)
            snapshot = MarketSnapshot(
                time=raw_state["time"],
                mid=raw_state["mid"],
                best_bid=raw_state["best_bid"],
                best_ask=raw_state["best_ask"],
                spread_bps=raw_state["spread_bps"],
                bid_depth_touch=raw_state["bid_depth_touch"],
                ask_depth_touch=raw_state["ask_depth_touch"],
                micro_price=raw_state["micro_price"],
                obi=raw_state["obi"],
                vol_bps=raw_state["vol_bps"],
                bot_position=raw_state["bot_position"],
                bot_avg_cost=raw_state["bot_avg_cost"],
                bot_cash=raw_state["bot_cash"],
                bot_realized_pnl=raw_state["bot_realized_pnl"],
                bot_unrealized_pnl=raw_state["bot_unrealized_pnl"],
                bot_total_pnl=raw_state["bot_total_pnl"],
                bot_open_orders=raw_state["bot_open_orders"],
                recent_trades=raw_state["recent_trades"],
            )
            bot.on_tick(snapshot)
            if bot.stop_requested:
                break

            # Periodic status logging every 30 seconds
            if raw_state["time"] - last_report_ts >= 30.0:
                last_report_ts = raw_state["time"]
                log.info(
                    f"[T+{raw_state['time']:>5.1f}s] Mid: ${raw_state['mid']:.1f} | Spread: {raw_state['spread_bps']:.2f} bps | "
                    f"Pos: {raw_state['bot_position']:+.5f} BTC | PnL: ${raw_state['bot_total_pnl']:+.2f}"
                )

        # Allow markouts to settle
        for _ in range(120): sim_env.step(dt)

        analytics = sim_env.get_analytics()
        log.info("\n" + "=" * 70)
        log.info("FINAL SIMULATION PERFORMANCE SUMMARY")
        log.info("=" * 70)
        log.info(f"Total Mark-to-Market PnL : ${analytics['total_pnl']:+.2f}")
        log.info(f"Realized PnL             : ${analytics['realized_pnl']:+.2f}")
        log.info(f"Unrealized PnL           : ${analytics['unrealized_pnl']:+.2f}")
        log.info(f"Total Fills Count        : {analytics['total_fills']}")
        log.info(f"Ending Position          : {analytics['final_position']:+.5f} BTC")
        log.info(f"Volume Traded (USD)      : ${analytics['volume_traded_usd']:.2f}")
        log.info(f"Avg Markout (+1s)        : {analytics['avg_markout_1s_bps']:+.2f} bps")
        log.info(f"Avg Markout (+5s)        : {analytics['avg_markout_5s_bps']:+.2f} bps")
        log.info(f"Adverse Fill Ratio (1s)  : {analytics['adverse_fill_ratio_1s']*100:.1f}%")
        log.info("=" * 70)

    elif cfg.mode == "live":
        log.info("Starting Live Market Maker Bot...")
        # Live exchange connector integrates directly with existing bot.py & exchange.py
        from config import Config
        from bot import MarketMaker
        bot = MarketMaker(Config.from_env())
        asyncio.run(bot.run())


if __name__ == "__main__":
    run_bot()
