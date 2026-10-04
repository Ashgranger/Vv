"""High-Fidelity Volatile Market Simulation Environment.

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
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from config import Config
from models import SimOrder, Trade, BotFill, MarketSnapshot


class VolatileMarketEnv:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()

        if self.cfg.sim_seed is not None:
            random.seed(self.cfg.sim_seed)

        self.initial_price = 80000.0
        self.mid = self.initial_price
        self.min_spread_bps = self.cfg.target_spread_min_bps
        self.max_spread_bps = self.cfg.target_spread_max_bps
        self.base_spread_bps = 2.5
        self.current_spread_bps = self.base_spread_bps
        self.tick_size = self.cfg.tick_size
        self.step_size = self.cfg.step_size

        self.noise_order_rate = self.cfg.sim_noise_rate
        self.informed_order_rate = self.cfg.sim_informed_rate
        self.jump_intensity = self.cfg.sim_jump_intensity
        self.base_vol_bps = self.cfg.sim_base_vol_bps
        self.current_vol_bps = self.base_vol_bps
        self.latency_s = self.cfg.sim_latency_s

        self.time = 0.0
        self.seq = 0

        # Bot orders and pending actions (latency queue)
        self.bot_orders: Dict[str, SimOrder] = {}
        self.pending_actions: List[Tuple[float, str, dict]] = []

        # Bot accounting
        self.bot_position = 0.0
        self.bot_avg_cost = 0.0
        self.bot_cash = 0.0
        self.bot_realized_pnl = 0.0
        self.bot_fees_paid = 0.0
        self.bot_volume_traded = 0.0
        self.bot_fills: List[BotFill] = []

        # History buffers
        self.mid_history: deque[Tuple[float, float]] = deque(maxlen=2000)
        self.trade_history: deque[Trade] = deque(maxlen=1000)
        self.pending_markouts: List[Tuple[float, BotFill, float]] = []

        # Order book state
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_depth_touch = 1.0
        self.ask_depth_touch = 1.0
        self.micro_price = self.initial_price
        self.obi = 0.0

        self._refresh_market_spread()
        self._rebuild_external_book()
        self.mid_history.append((self.time, self.mid))

    def _round_tick(self, px: float) -> float:
        return round(round(px / self.tick_size) * self.tick_size, 4)

    def _round_step(self, qty: float) -> float:
        return round(round(qty / self.step_size) * self.step_size, 6)

    def _refresh_market_spread(self) -> None:
        """Dynamically updates the market spread between 1 and 8 bps based on volatility and impact."""
        vol_ratio = self.current_vol_bps / self.base_vol_bps
        spread_target = self.base_spread_bps * (0.6 + 0.5 * vol_ratio)
        noise = random.gauss(0.0, 0.4)
        raw_spread = self.current_spread_bps * 0.85 + (spread_target + noise) * 0.15
        self.current_spread_bps = max(self.min_spread_bps, min(self.max_spread_bps, raw_spread))

    def _rebuild_external_book(self) -> None:
        """Regenerates external liquidity levels conforming to the 1-8 bps spread."""
        half_spread = (self.mid * (self.current_spread_bps / 10000.0)) / 2.0
        target_bid = self._round_tick(self.mid - half_spread)
        target_ask = self._round_tick(self.mid + half_spread)

        if target_ask <= target_bid:
            target_ask = self._round_tick(target_bid + self.tick_size)

        self.best_bid = target_bid
        self.best_ask = target_ask

        # Dynamic depths proportional to market activity
        self.bid_depth_touch = max(0.005, round(random.uniform(0.01, 0.03), 4))
        self.ask_depth_touch = max(0.005, round(random.uniform(0.01, 0.03), 4))

        total_depth = self.bid_depth_touch + self.ask_depth_touch
        self.micro_price = (
            (self.best_bid * self.ask_depth_touch + self.best_ask * self.bid_depth_touch)
            / total_depth
        )
        self.obi = (self.bid_depth_touch - self.ask_depth_touch) / total_depth

    # ---- Bot Interaction API ---------------------------------------------- #

    def place_order(self, side: str, price: float, qty: float) -> str:
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
        exec_time = self.time + self.latency_s
        self.pending_actions.append((exec_time, "CANCEL", {"order_id": order_id}))

    def cancel_all_orders(self) -> None:
        exec_time = self.time + self.latency_s
        self.pending_actions.append((exec_time, "CANCEL_ALL", {}))

    def _execute_place(self, payload: dict) -> None:
        order_id = payload["order_id"]
        side = payload["side"]
        price = payload["price"]
        qty = payload["qty"]

        # Post-Only Check (ALO)
        if side == "BUY" and price >= self.best_ask: return
        if side == "SELL" and price <= self.best_bid: return

        queue_ahead = 0.0
        if side == "BUY":
            if price == self.best_bid:
                queue_ahead = self.bid_depth_touch
            elif price < self.best_bid:
                ticks_back = (self.best_bid - price) / self.tick_size
                queue_ahead = self.bid_depth_touch + ticks_back * 0.05
        else:
            if price == self.best_ask:
                queue_ahead = self.ask_depth_touch
            elif price > self.best_ask:
                ticks_back = (price - self.best_ask) / self.tick_size
                queue_ahead = self.ask_depth_touch + ticks_back * 0.05

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

    # ---- Matching & Fill Engine ------------------------------------------- #

    def _process_market_order(self, side: str, qty: float, is_informed: bool) -> None:
        trade_px = self.best_ask if side == "BUY" else self.best_bid
        bot_fill_qty = 0.0
        matched_bot_order_id = None

        for oid, o in list(self.bot_orders.items()):
            if not o.active: continue

            if side == "BUY" and o.side == "SELL":
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

        self.trade_history.append(Trade(
            timestamp=self.time,
            side=side,
            price=trade_px,
            qty=qty,
            is_informed=is_informed,
            bot_fill_qty=bot_fill_qty,
            bot_order_id=matched_bot_order_id
        ))

        if side == "BUY":
            self.ask_depth_touch = max(0.005, self.ask_depth_touch - qty)
            if self.ask_depth_touch <= 0.005:
                self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 0.8)
        else:
            self.bid_depth_touch = max(0.005, self.bid_depth_touch - qty)
            if self.bid_depth_touch <= 0.005:
                self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 0.8)

    def _record_bot_fill(self, order: SimOrder, fill_qty: float, fill_price: float) -> None:
        signed_qty = fill_qty if order.side == "BUY" else -fill_qty
        prev_pos = self.bot_position
        new_pos = prev_pos + signed_qty

        realized_delta = 0.0
        if prev_pos == 0.0 or (prev_pos > 0) == (signed_qty > 0):
            total_abs = abs(prev_pos) + fill_qty
            self.bot_avg_cost = (self.bot_avg_cost * abs(prev_pos) + fill_price * fill_qty) / total_abs
        else:
            closed_qty = min(abs(prev_pos), fill_qty)
            direction = 1.0 if prev_pos > 0 else -1.0
            realized_delta = (fill_price - self.bot_avg_cost) * closed_qty * direction
            if abs(new_pos) < 1e-8:
                self.bot_avg_cost = 0.0
            elif (new_pos > 0) != (prev_pos > 0):
                self.bot_avg_cost = fill_price

        fee = fill_qty * fill_price * 0.0  # Zero maker fee
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

        self.pending_markouts.append((self.time + 1.0, fill_record, 1.0))
        self.pending_markouts.append((self.time + 5.0, fill_record, 5.0))

    def _process_markouts(self) -> None:
        rem = []
        for target_time, fill, horizon in self.pending_markouts:
            if self.time >= target_time:
                if fill.side == "BUY":
                    m_bps = (self.mid - fill.price) / fill.price * 10000.0
                else:
                    m_bps = (fill.price - self.mid) / fill.price * 10000.0

                if horizon == 1.0: fill.markout_1s = m_bps
                elif horizon == 5.0: fill.markout_5s = m_bps
            else:
                rem.append((target_time, fill, horizon))
        self.pending_markouts = rem

    # ---- Environment Step ------------------------------------------------- #

    def step(self, dt: float = 0.05) -> MarketSnapshot:
        self.time += dt

        # 1. Execute latency queue actions
        ready = [act for act in self.pending_actions if act[0] <= self.time]
        self.pending_actions = [act for act in self.pending_actions if act[0] > self.time]
        for _, act_type, payload in ready:
            if act_type == "PLACE": self._execute_place(payload)
            elif act_type == "CANCEL": self._execute_cancel(payload)
            elif act_type == "CANCEL_ALL": self._execute_cancel_all()

        # 2. Stochastic Volatility Update
        vol_drift = 1.2 * (self.base_vol_bps - self.current_vol_bps) * dt
        vol_shock = random.gauss(0, 1.5) * math.sqrt(dt)
        self.current_vol_bps = max(5.0, self.current_vol_bps + vol_drift + vol_shock)

        # 3. Mid-Price Simulation (GBM + Compound Poisson Jumps)
        sigma = (self.current_vol_bps / 10000.0)
        diffusion = sigma * math.sqrt(dt) * random.gauss(0, 1)

        jump = 0.0
        jump_occurred = False
        jump_side = None
        if random.random() < self.jump_intensity * dt:
            jump_occurred = True
            jump_mag_bps = random.uniform(2.5, 9.0)
            jump_sign = random.choice([1.0, -1.0])
            jump = jump_sign * (jump_mag_bps / 10000.0)
            jump_side = "BUY" if jump_sign > 0 else "SELL"
            self.current_vol_bps += jump_mag_bps * 1.5
            self.current_spread_bps = min(self.max_spread_bps, self.current_spread_bps + 2.0)

        self.mid = self.mid * math.exp(diffusion + jump)
        self.mid_history.append((self.time, self.mid))

        self._refresh_market_spread()
        self._rebuild_external_book()

        # 4. Market Order Arrivals
        if jump_occurred:
            informed_size = random.uniform(0.02, 0.06)
            self._process_market_order(side=jump_side, qty=informed_size, is_informed=True)
        elif random.random() < self.informed_order_rate * dt:
            recent_ret = (self.mid - self.mid_history[0][1]) / self.mid_history[0][1] * 10000.0 if len(self.mid_history) > 1 else 0.0
            if abs(recent_ret) > 1.5:
                side = "BUY" if recent_ret > 0 else "SELL"
                inf_qty = random.uniform(0.005, 0.025)
                self._process_market_order(side=side, qty=inf_qty, is_informed=True)

        num_noise_orders = int(random.expovariate(1.0 / max(0.01, self.noise_order_rate * dt))) if random.random() < 0.8 else 0
        for _ in range(min(num_noise_orders, 4)):
            side = random.choice(["BUY", "SELL"])
            noise_qty = random.uniform(0.005, 0.025)
            self._process_market_order(side=side, qty=noise_qty, is_informed=False)

        # 5. Queue decay
        for o in self.bot_orders.values():
            if o.active and o.queue_ahead > 0:
                o.queue_ahead = max(0.0, o.queue_ahead - o.queue_ahead * 0.15 * dt)

        self._process_markouts()
        return self.get_snapshot()

    def get_snapshot(self) -> MarketSnapshot:
        unrealized = (self.mid - self.bot_avg_cost) * self.bot_position if self.bot_position != 0 else 0.0
        total_pnl = self.bot_realized_pnl + unrealized

        return MarketSnapshot(
            time=self.time,
            mid=self.mid,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            spread_bps=self.current_spread_bps,
            bid_depth_touch=self.bid_depth_touch,
            ask_depth_touch=self.ask_depth_touch,
            micro_price=self.micro_price,
            obi=self.obi,
            vol_bps=self.current_vol_bps,
            bot_position=self.bot_position,
            bot_avg_cost=self.bot_avg_cost,
            bot_cash=self.bot_cash,
            bot_realized_pnl=self.bot_realized_pnl,
            bot_unrealized_pnl=unrealized,
            bot_total_pnl=total_pnl,
            bot_open_orders={
                oid: {"side": o.side, "price": o.price, "qty": o.remaining, "queue_ahead": o.queue_ahead}
                for oid, o in self.bot_orders.items() if o.active
            },
            recent_trades=list(self.trade_history)[-20:],
        )

    def get_analytics(self) -> dict:
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
