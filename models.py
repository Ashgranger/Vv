"""Data structures, domain models, and quantitative helper functions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional


# ---- Quantitative Constants & Helpers ------------------------------------- #

BPS = Decimal("10000")
ZERO = Decimal("0")
ONE = Decimal("1")


def clamp(val: float | Decimal, low: float | Decimal, high: float | Decimal) -> Any:
    return max(low, min(high, val))


def q_down(val: float | Decimal, step: float | Decimal) -> Any:
    """Quantizes down to nearest step / tick."""
    if isinstance(val, Decimal) and isinstance(step, Decimal):
        return (val // step) * step
    return round(math.floor(float(val) / float(step)) * float(step), 6)


def q_up(val: float | Decimal, step: float | Decimal) -> Any:
    """Quantizes up to nearest step / tick."""
    if isinstance(val, Decimal) and isinstance(step, Decimal):
        return math.ceil(val / step) * step
    return round(math.ceil(float(val) / float(step)) * float(step), 6)


def fmt(val: float | Decimal | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{float(val):.{decimals}f}"


# ---- Domain Data Models --------------------------------------------------- #

@dataclass
class Market:
    market_id: int
    name: str
    status: str
    tick: float
    step: float
    min_notional: float = 5.0
    min_size: float = 0.0001
    max_size: float = 100.0


@dataclass
class SimOrder:
    order_id: str
    owner: str             # "bot" or "external"
    side: str              # "BUY" or "SELL"
    price: float
    qty: float
    remaining: float
    queue_ahead: float     # Volume ahead in the matching queue
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
    recent_trades: List[Trade]


@dataclass
class QuoteTarget:
    side: str             # "BUY" or "SELL"
    price: float
    qty: float
    is_exit_quote: bool   # True if reducing / unwinding position
