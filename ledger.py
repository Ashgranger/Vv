"""Fill-driven accounting + Level 6 Online Learning of Adverse Selection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

from utils import BPS, BUY, SELL, ZERO


@dataclass
class Fill:
    ts: float
    side: str
    qty: Decimal
    price: Decimal
    mid: Decimal
    edge_bps: Decimal
    position: Decimal
    realized_delta: Decimal


class Ledger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = ZERO
        self.avg_cost = ZERO
        self.realized = ZERO
        self.fees = ZERO
        self.spread_capture = ZERO
        self.spread_edge_bps_sum = ZERO
        self.volume_usd = ZERO
        self.n_fills = 0
        self.n_buys = 0
        self.n_sells = 0
        self.fills: deque = deque(maxlen=500)
        self.opened_ts: Optional[float] = None
        self.last_fill_ts = -1e9
        
        self._pending_markouts: list = []
        self.markouts: deque = deque(maxlen=cfg.markout_window * 2)
        self.markouts_buy: deque = deque(maxlen=cfg.markout_window)
        self.markouts_sell: deque = deque(maxlen=cfg.markout_window)
        self._mismatch = 0

    def is_flat(self, mid: Decimal, min_notional: Decimal) -> bool:
        return abs(self.position * mid) < max(min_notional, Decimal(1))

    def hold_s(self, now: float) -> float:
        return now - self.opened_ts if self.opened_ts is not None else 0.0

    def unrealized(self, mark: Decimal) -> Decimal:
        return (mark - self.avg_cost) * self.position if self.position != 0 else ZERO

    def total_pnl(self, mark: Decimal) -> Decimal:
        return self.realized + self.unrealized(mark)

    def inventory_pnl(self, mark: Decimal) -> Decimal:
        return self.total_pnl(mark) - self.spread_capture

    @property
    def avg_edge_bps(self) -> Decimal:
        return (self.spread_edge_bps_sum / self.n_fills) if self.n_fills else ZERO

    def on_fill(self, side: str, qty: Decimal, price: Decimal, mid: Decimal, now: float,
                min_notional: Decimal) -> Fill:
        signed = qty if side == BUY else -qty
        was_flat = self.is_flat(mid, min_notional)
        realized_delta = ZERO
        if self.position == 0 or (self.position > 0) == (signed > 0):
            total = abs(self.position) + qty
            self.avg_cost = (self.avg_cost * abs(self.position) + price * qty) / total
            self.position += signed
        else:
            closed = min(abs(self.position), qty)
            direction = 1 if self.position > 0 else -1
            realized_delta = (price - self.avg_cost) * closed * direction
            new_pos = self.position + signed
            if new_pos == 0:
                self.avg_cost = ZERO
            elif (new_pos > 0) != (self.position > 0):
                self.avg_cost = price
            self.position = new_pos
        fee = qty * price * self.cfg.maker_fee_bps / BPS
        realized_delta -= fee
        self.fees += fee
        self.realized += realized_delta

        edge = (mid - price) if side == BUY else (price - mid)
        edge_bps = edge / mid * BPS if mid else ZERO
        self.spread_capture += edge * qty
        self.spread_edge_bps_sum += edge_bps
        self.volume_usd += qty * price
        self.n_fills += 1
        self.n_buys += (side == BUY)
        self.n_sells += (side != BUY)
        self.last_fill_ts = now

        now_flat = self.is_flat(mid, min_notional)
        if was_flat and not now_flat:
            self.opened_ts = now
        elif now_flat:
            self.opened_ts = None

        f = Fill(now, side, qty, price, mid, edge_bps, self.position, realized_delta)
        self.fills.append(f)
        self._pending_markouts.append((now + self.cfg.markout_horizon_s, f, self.cfg.markout_horizon_s))
        return f

    def process_markouts(self, mid: Decimal, now: float) -> None:
        while self._pending_markouts and self._pending_markouts[0][0] <= now:
            _, f, horizon = self._pending_markouts.pop(0)
            m = (mid - f.price) if f.side == BUY else (f.price - mid)
            m_bps = m / f.price * BPS
            self.markouts.append(m_bps)
            if f.side == BUY:
                self.markouts_buy.append(m_bps)
            else:
                self.markouts_sell.append(m_bps)

    @property
    def avg_markout_bps(self) -> Decimal:
        return sum(self.markouts, ZERO) / len(self.markouts) if self.markouts else ZERO

    @property
    def tox_bps(self) -> Decimal:
        if not self.markouts:
            return ZERO
        return max(ZERO, -self.avg_markout_bps)

    def side_tox_bps(self, side: str) -> Decimal:
        buf = self.markouts_buy if side == BUY else self.markouts_sell
        if not buf:
            return self.tox_bps
        avg_m = sum(buf, ZERO) / len(buf)
        return max(ZERO, -avg_m)

    def reconcile(self, ex_pos: Decimal, now: float, mid: Decimal, min_notional: Decimal) -> bool:
        tol = (min_notional / mid) * Decimal("0.25") if mid else Decimal("1e-8")
        if abs(ex_pos - self.position) <= tol:
            self._mismatch = 0
            return False
        if now - self.last_fill_ts < 4.0:
            return False
        self._mismatch += 1
        if self._mismatch < 2:
            return False
        old = self.position
        self.position = ex_pos
        self._mismatch = 0
        if old == 0 or (old > 0) != (ex_pos > 0) or self.avg_cost == 0:
            self.avg_cost = mid
        self.opened_ts = None if self.is_flat(mid, min_notional) else now
        return True
