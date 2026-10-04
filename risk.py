"""Adverse Selection Guards and Risk Management.

Implements:
1. Momentum Knife Filter (Falling knife / freight train protection)
2. Consecutive Fill Burst Cooldown
3. Queue Depletion & Thinning Fade
4. Session Loss Circuit Breaker
5. Inventory Stress & Liquidation Trigger
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

from config import Config
from models import MarketSnapshot, BotFill

log = logging.getLogger("VolatileMM.risk")


class RiskGuard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mid_buffer: deque[Tuple[float, float]] = deque(maxlen=200)
        self.cooldown_until: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}
        self.recent_fills: deque[Tuple[float, str]] = deque(maxlen=20)
        self.last_knife_ts: Dict[str, float] = {"BUY": -100.0, "SELL": -100.0}

    def register_mid(self, now: float, mid: float) -> None:
        self.mid_buffer.append((now, mid))

    def on_fill(self, fill: BotFill, now: float) -> None:
        """Processes fills and evaluates consecutive burst triggers."""
        self.recent_fills.append((now, fill.side))
        recent_same = [f for f in self.recent_fills if f[1] == fill.side and now - f[0] <= self.cfg.burst_window_s]
        if len(recent_same) >= self.cfg.burst_threshold:
            self.cooldown_until[fill.side] = now + self.cfg.burst_cooldown_s
            log.warning(f"BURST GUARD TRIGGERED: Pausing {fill.side} side for {self.cfg.burst_cooldown_s}s")

    def evaluate_guards(self, state: MarketSnapshot) -> Tuple[bool, bool, float]:
        """Evaluates whether to suppress quoting on exposed sides."""
        now = state.time
        buy_blocked = (now < self.cooldown_until["BUY"])
        sell_blocked = (now < self.cooldown_until["SELL"])

        # 1. Momentum / Falling Knife Filter (with anti-hysteresis)
        ret_trend_bps = 0.0
        if len(self.mid_buffer) >= 2:
            target_ts = now - self.cfg.trend_window_s
            past_mid = self.mid_buffer[0][1]
            for ts, px in self.mid_buffer:
                if ts >= target_ts:
                    past_mid = px
                    break
            ret_trend_bps = (state.mid - past_mid) / past_mid * 10000.0

            if ret_trend_bps <= -self.cfg.trend_pull_bps:
                buy_blocked = True
            elif ret_trend_bps >= self.cfg.trend_pull_bps:
                sell_blocked = True

        # 2. Hard Capacity Constraints
        if state.bot_position >= self.cfg.max_position - 1e-6:
            buy_blocked = True
        if state.bot_position <= -self.cfg.max_position + 1e-6:
            sell_blocked = True

        # 3. Calculate directional flow pressure
        recent_trades = state.recent_trades
        t_2s = [t for t in recent_trades if now - t.timestamp <= 2.0]
        tfi = 0.0
        if t_2s:
            b_vol = sum(t.qty for t in t_2s if t.side == "BUY")
            s_vol = sum(t.qty for t in t_2s if t.side == "SELL")
            tot = b_vol + s_vol
            if tot > 0: tfi = (b_vol - s_vol) / tot

        dir_pressure_bps = (0.8 * tfi + 0.5 * state.obi + 0.3 * (ret_trend_bps / 2.0)) * self.cfg.adverse_fade_mult

        return buy_blocked, sell_blocked, dir_pressure_bps

    def is_stressed(self, state: MarketSnapshot, position_opened_ts: Optional[float]) -> bool:
        """Determines whether the position is under stress and requires touch liquidation."""
        if state.bot_position == 0.0:
            return False

        now = state.time
        holding_time = (now - position_opened_ts) if position_opened_ts else 0.0
        unrealized_bps = (
            (state.mid - state.bot_avg_cost) / state.bot_avg_cost * 10000.0
            if state.bot_position > 0
            else (state.bot_avg_cost - state.mid) / state.bot_avg_cost * 10000.0
        )

        return (
            holding_time > self.cfg.max_hold_s
            or unrealized_bps <= -self.cfg.stress_loss_bps
            or abs(state.bot_position) >= self.cfg.soft_position
        )

    def check_circuit_breaker(self, state: MarketSnapshot) -> bool:
        """Checks if session max loss has been breached."""
        if state.bot_total_pnl <= -self.cfg.session_max_loss_usd:
            log.error(f"CIRCUIT BREAKER: Session loss (${state.bot_total_pnl:.2f}) <= -${self.cfg.session_max_loss_usd}")
            return True
        return False
