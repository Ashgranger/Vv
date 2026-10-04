"""Accounting, Position Tracking, and Adverse Selection Markout Measurement."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from config import Config
from models import BotFill


class AccountingTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position = 0.0
        self.avg_cost = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.volume_traded_usd = 0.0
        self.position_opened_ts: Optional[float] = None

        self.fills: List[BotFill] = []
        self.pending_markouts: List[Tuple[float, BotFill, float]] = []

        # Markout history buffers for online toxicity learning
        self.markouts_buy: deque[float] = deque(maxlen=20)
        self.markouts_sell: deque[float] = deque(maxlen=20)

    def record_fill(
        self,
        order_id: str,
        side: str,
        price: float,
        qty: float,
        mid: float,
        now: float,
        maker_fee_bps: float = 0.0,
    ) -> BotFill:
        """Processes a fill and updates inventory, cash, and PnL."""
        signed_qty = qty if side == "BUY" else -qty
        prev_pos = self.position
        new_pos = prev_pos + signed_qty

        realized_delta = 0.0
        if prev_pos == 0.0 or (prev_pos > 0) == (signed_qty > 0):
            # Increasing position
            total_abs = abs(prev_pos) + qty
            self.avg_cost = (self.avg_cost * abs(prev_pos) + price * qty) / total_abs
            if prev_pos == 0.0:
                self.position_opened_ts = now
        else:
            # Closing / unwinding position
            closed_qty = min(abs(prev_pos), qty)
            direction = 1.0 if prev_pos > 0 else -1.0
            realized_delta = (price - self.avg_cost) * closed_qty * direction
            if abs(new_pos) < 1e-8:
                self.avg_cost = 0.0
                self.position_opened_ts = None
            elif (new_pos > 0) != (prev_pos > 0):
                self.avg_cost = price
                self.position_opened_ts = now

        fee = qty * price * (maker_fee_bps / 10000.0)
        realized_delta -= fee
        self.fees_paid += fee
        self.realized_pnl += realized_delta
        self.cash -= (signed_qty * price + fee)
        self.position = new_pos
        self.volume_traded_usd += qty * price

        fill_record = BotFill(
            timestamp=now,
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            mid_at_fill=mid,
            position_after=new_pos,
            realized_pnl_delta=realized_delta,
            fee_paid=fee,
        )
        self.fills.append(fill_record)

        # Schedule markout checks at 1.0s and 5.0s horizons
        self.pending_markouts.append((now + 1.0, fill_record, 1.0))
        self.pending_markouts.append((now + 5.0, fill_record, 5.0))
        return fill_record

    def process_markouts(self, mid: float, now: float) -> None:
        """Evaluates matured post-fill markouts to measure adverse selection."""
        rem = []
        for target_time, fill, horizon in self.pending_markouts:
            if now >= target_time:
                # Markout in bps: (+ = favorable price move, - = adverse selection)
                if fill.side == "BUY":
                    m_bps = (mid - fill.price) / fill.price * 10000.0
                else:
                    m_bps = (fill.price - mid) / fill.price * 10000.0

                if horizon == 1.0:
                    fill.markout_1s = m_bps
                    if fill.side == "BUY": self.markouts_buy.append(m_bps)
                    else: self.markouts_sell.append(m_bps)
                elif horizon == 5.0:
                    fill.markout_5s = m_bps
            else:
                rem.append((target_time, fill, horizon))
        self.pending_markouts = rem

    def get_unrealized_pnl(self, mid: float) -> float:
        if self.position == 0.0:
            return 0.0
        return (mid - self.avg_cost) * self.position

    def get_total_pnl(self, mid: float) -> float:
        return self.realized_pnl + self.get_unrealized_pnl(mid)

    def get_toxicity_bps(self) -> Tuple[float, float]:
        """Returns side-specific toxicity penalty in bps."""
        buy_tox = max(0.0, -sum(self.markouts_buy) / len(self.markouts_buy)) if self.markouts_buy else 0.0
        sell_tox = max(0.0, -sum(self.markouts_sell) / len(self.markouts_sell)) if self.markouts_sell else 0.0
        return buy_tox, sell_tox
