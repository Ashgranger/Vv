"""Quote engine. Pure function of a Snapshot -> Plan (no I/O), so it is unit-testable.

How a professional-style maker quotes, in short:
  * fair value  = micro price (size-weighted mid) - the thin side of the book is where price goes
  * edge        = distance from fair value we insist on: max(MIN_EDGE, fee + toxicity + k*vol)
  * inventory   = skew fair value against the position, but NEVER cross the book (a post-only quote
                  that would cross is rejected - that is what trapped inventory in the v1 run)
  * placement   = sit AT the touch (or 1 tick inside it when the spread is wide) whenever our edge
                  allows it - queue position is the product - otherwise sit behind it at our edge
  * exits       = reduce-side quote never worse than cost + a small profit, until 'stress'
  * guards      = only the side that ADDS exposure is ever pulled/widened; the reducing side stays
                  live so we can always get out
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from market import Market
from utils import BPS, BUY, SELL, ZERO, ONE, Fatal, bps_diff, clamp, q_down, q_up


@dataclass
class Snapshot:
    now: float
    market: Market
    bid: Decimal
    ask: Decimal
    mid: Decimal
    micro: Decimal
    position: Decimal
    avg_cost: Decimal
    hold_s: float
    ret_bps: Decimal            # signed short-window return (negative = falling)
    move_bps: Decimal           # short-window high-low range
    vol_bps: Decimal            # smoothed range
    tox_bps: Decimal            # recent negative markout (adverse selection)
    imbalance: Decimal = ZERO   # signed book depth pressure, -1..1 (see MarketData.depth_imbalance);
                                # positive = more bid depth = buying pressure = price likely to tick up
    cooldown_until: dict = field(default_factory=lambda: {BUY: 0.0, SELL: 0.0})
    jump_active: bool = False
    halted: bool = False


@dataclass
class Target:
    price: Decimal
    qty: Decimal
    role: str                   # "add" | "reduce"


@dataclass
class Plan:
    bid: Optional[Target]
    ask: Optional[Target]
    edge_bps: Decimal = ZERO
    skew_bps: Decimal = ZERO
    stress: bool = False
    notes: list = field(default_factory=list)
    # side -> guard code ("trend" | "vol" | "jump" | "burst" | None) for the ADDING side only,
    # structured (not parsed from `notes`) so the bot can arm a hysteresis cooldown on "trend".
    blocked: dict = field(default_factory=lambda: {BUY: None, SELL: None})
    # extra "add"-only levels beyond the touch quote, nearest-first (see Config.extra_levels).
    # Empty on a "reduce" (exit) side, on a guard-blocked side, or when EXTRA_LEVELS=0.
    extra_bids: list = field(default_factory=list)
    extra_asks: list = field(default_factory=list)


class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, s: Snapshot) -> Plan:
        c, m = self.cfg, s.market
        notes: list = []
        tb, ta = m.tick_for(s.bid), m.tick_for(s.ask)
        pos_usd = s.position * s.mid
        flat = abs(pos_usd) < max(m.min_notional, Decimal(1))
        long_ = (not flat) and s.position > 0
        short_ = (not flat) and s.position < 0

        # ---- stress: position is hurting / stale -> stop adding, get out at the touch ------
        stress = s.halted
        if s.halted:
            notes.append("HALTED")
        if not flat and s.avg_cost > 0:
            mark = s.bid if long_ else s.ask                       # what we could actually get
            pnl_bps = bps_diff(mark, s.avg_cost) * (1 if long_ else -1)
            if pnl_bps <= -c.stress_loss_bps:
                stress = True
                notes.append(f"stress: position {pnl_bps:.1f}bps underwater")
            if s.hold_s >= c.max_hold_s:
                stress = True
                notes.append(f"stress: held {s.hold_s:.0f}s")

        # ---- edge & fair value --------------------------------------------------------
        edge = max(c.min_edge_bps, c.maker_fee_bps + c.tox_mult * s.tox_bps + c.vol_k * s.vol_bps)
        edge = min(edge, c.max_edge_bps)
        if s.tox_bps > 0:
            notes.append(f"toxic fills: +{c.tox_mult * s.tox_bps:.1f}bps edge")
        inv_ratio = clamp(pos_usd / c.max_position_usd, -ONE, ONE)
        skew = c.skew_bps * inv_ratio
        ref = s.micro if c.use_micro else s.mid
        res = ref * (1 - skew / BPS)                                # long -> both quotes shift down

        bid_role = "reduce" if short_ else "add"
        ask_role = "reduce" if long_ else "add"
        # Book pressure (leading) on top of realized trend (lagging). Positive imbalance = buying
        # pressure = price likely up = the ASK is the side about to be run over; negative = the BID.
        # Same rule as trend_widen: only ever the ADDING side - the exit side is never widened/shrunk.
        imb = clamp(s.imbalance, -ONE, ONE) if c.use_depth_imbalance else ZERO
        bid_danger, ask_danger = max(ZERO, -imb), max(ZERO, imb)
        bid_imb_mult = ONE - c.imbalance_size_cut * bid_danger if bid_role == "add" else ONE
        ask_imb_mult = ONE - c.imbalance_size_cut * ask_danger if ask_role == "add" else ONE
        bid_extra = (max(ZERO, -s.ret_bps) * c.trend_widen + bid_danger * c.imbalance_widen_bps) if bid_role == "add" else ZERO
        ask_extra = (max(ZERO, s.ret_bps) * c.trend_widen + ask_danger * c.imbalance_widen_bps) if ask_role == "add" else ZERO
        if c.use_depth_imbalance and abs(imb) >= Decimal("0.4"):
            notes.append(f"book pressure {imb:+.2f} -> {'ask' if imb > 0 else 'bid'} widened/shrunk")
        bid_cap = res * (1 - (edge + bid_extra) / BPS)              # highest bid we accept
        ask_floor = res * (1 + (edge + ask_extra) / BPS)            # lowest ask we accept

        # ---- placement relative to the book (never cross) ----------------------------------
        spread = s.ask - s.bid
        penny = c.penny and spread >= 2 * max(tb, ta)
        touch_bid = s.bid + (tb if penny else ZERO)
        touch_ask = s.ask - (ta if penny else ZERO)
        bid_px = min(bid_cap, touch_bid)
        ask_px = max(ask_floor, touch_ask)
        if c.aggressive_touch:               # join the actual touch outright, ignore min_edge_bps
            bid_px, ask_px = touch_bid, touch_ask  # (exit-floor below still protects the reduce side)

        if long_:
            ask_px = touch_ask if stress else max(ask_px, s.avg_cost * (1 + c.exit_min_profit_bps / BPS))
        if short_:
            bid_px = touch_bid if stress else min(bid_px, s.avg_cost * (1 - c.exit_min_profit_bps / BPS))

        bid_px = q_down(min(bid_px, s.ask - tb), tb)
        ask_px = q_up(max(ask_px, s.bid + ta), ta)
        if bid_px >= ask_px:
            return Plan(None, None, edge, skew, stress, notes + ["book too tight after rounding"])

        # ---- sizes -------------------------------------------------------------------------
        base = q_down(c.order_usd / s.mid, m.step)
        if base < m.min_size or base * s.mid < m.min_notional or (m.max_size and base > m.max_size):
            raise Fatal(f"ORDER_USD={c.order_usd} -> qty {base}; market needs size>={m.min_size}, "
                        f"notional>={m.min_notional}, size<={m.max_size}")

        def sized_ok(q: Decimal, px: Decimal) -> bool:
            return q >= m.min_size and q * px >= m.min_notional and (not m.max_size or q <= m.max_size)

        def reduce_qty() -> Decimal:
            q = abs(s.position) if stress else min(abs(s.position), base)
            return q_down(q, m.step)

        def add_qty(same_dir: bool, side_mult: Decimal = ONE) -> Decimal:
            scale = ONE - Decimal("0.5") * abs(inv_ratio) if same_dir else ONE   # smaller when loaded
            return q_down(base * scale * side_mult, m.step)                      # smaller into book pressure

        def blocked(side: str) -> tuple:
            """Returns (human_text, code). code is None when not blocked, else one of
            "stress"/"burst"/"jump"/"vol"/"trend" - used by the bot to arm hysteresis."""
            if stress:
                return "stress", "stress"
            if s.cooldown_until.get(side, 0.0) > s.now:
                return "fill-burst cooldown", "burst"
            if s.jump_active:
                return "price jump", "jump"
            if s.move_bps >= c.vol_pause_bps:
                return f"vol pause ({s.move_bps:.1f}bps range)", "vol"
            if side == BUY and s.ret_bps <= -c.trend_pull_bps:
                return f"falling {s.ret_bps:.1f}bps", "trend"
            if side == SELL and s.ret_bps >= c.trend_pull_bps:
                return f"rising {s.ret_bps:.1f}bps", "trend"
            return None, None

        blocked_map = {BUY: None, SELL: None}
        bid = ask = None
        extra_bids: list = []
        extra_asks: list = []

        def ladder(side: str, px0: Decimal, tick: Decimal, extra0: Decimal, role: str, qty_fn,
                   start_i: int = 1) -> list:
            """Extra levels beyond px0, each further from fair value than the last. qty_fn(i, px_i)
            returns the qty for level i, or None to stop laddering there (used for both the
            position-cap check on 'add' levels and the remaining-position check on 'reduce'
            levels - see the two closures below). Stops the moment a level would round onto/past
            the previous one, fail sizing, or qty_fn says stop. start_i lets a caller resume the
            same edge/spacing progression after another ladder() call already used levels 1..start_i-1
            (see 'continue past the reduce ladder' below) - i keeps meaning the same fixed distance
            from fair value everywhere, regardless of which call produced it."""
            out: list = []
            prev = px0
            sign = 1 if side == BUY else -1
            for i in range(start_i, c.extra_levels + 1):
                lvl_edge = edge + extra0 + i * c.level_spacing_bps
                raw = res * (1 - sign * lvl_edge / BPS)
                px_i = q_down(raw, tick) if side == BUY else q_up(raw, tick)
                if (side == BUY and px_i >= prev) or (side == SELL and px_i <= prev):
                    break                                    # tick rounding collapsed the level
                qty_i = qty_fn(i, px_i)
                if qty_i is None or not sized_ok(qty_i, px_i):
                    break
                out.append(Target(px_i, qty_i, role))
                prev = px_i
            return out

        def continue_adding(side: str, extra: list, px0: Decimal, tick: Decimal) -> list:
            """Once the reduce ladder above has claimed only as many levels as the CURRENT position
            actually needs (often just level 0, for a small residual), the rest of the EXTRA_LEVELS
            budget doesn't have to sit idle: use it for fresh 'add' quotes, so a bad fill's exit stays
            fully protected (untouched - these levels only ever get appended after it, never priced or
            sized off it) while the bot keeps capturing spread in the meantime. Position-capped exactly
            like the normal add ladder, starting from a flat baseline: these levels sit beyond every
            reduce level in price, so - barring a gap that fills the whole ladder at once - they'd only
            themselves fill after the reduce levels already had, i.e. after the position they're
            protecting was already closed out. Subject to the SAME guards as normal add quoting
            (unlike the reduce order itself, which - by existing design - stays live through every
            guard so the bot can always get out); and never while in stress/halted, full stop - that's
            exactly when only the exit should be working."""
            used = len(extra)
            if not c.continue_add_after_reduce or used >= c.extra_levels or stress:
                return extra
            why, code = blocked(side)
            blocked_map[side] = code
            if why:
                notes.append(f"no continued {'bid' if side == BUY else 'ask'}: {why}")
                return extra
            running = [ZERO]

            def add_q(i: int, px_i: Decimal, r=running) -> Optional[Decimal]:
                qty_i = q_down(add_qty(False, bid_imb_mult if side == BUY else ask_imb_mult) * (c.level_size_mult ** i), m.step)   # fresh from an
                if qty_i <= 0:                                                      # assumed-flat baseline
                    return None
                sign = 1 if side == BUY else -1
                new_running = r[0] + sign * qty_i * px_i
                if abs(new_running) > c.max_position_usd:
                    return None
                r[0] = new_running
                return qty_i
            more = ladder(side, px0, tick, ZERO, "add", add_q, start_i=used + 1)
            return extra + more

        if bid_role == "reduce":
            q = reduce_qty()
            if sized_ok(q, bid_px):
                bid = Target(bid_px, q, "reduce")
            if c.extra_levels > 0:
                remaining = [abs(s.position) - (q if bid else ZERO)]

                def reduce_q(i: int, px_i: Decimal, r=remaining) -> Optional[Decimal]:
                    if r[0] <= 0:
                        return None
                    take = q_down(min(r[0], base * (c.level_size_mult ** i)), m.step)
                    if take <= 0:
                        return None
                    r[0] -= take
                    return take
                extra_bids = ladder(BUY, bid_px, tb, bid_extra, "reduce", reduce_q)
                start_px = extra_bids[-1].price if extra_bids else bid_px
                extra_bids = continue_adding(BUY, extra_bids, start_px, tb)
        else:
            why, code = blocked(BUY)
            blocked_map[BUY] = code
            q = add_qty(long_, bid_imb_mult)
            if why:
                notes.append(f"no bid: {why}")
            elif not sized_ok(q, bid_px):
                notes.append("no bid: size below minimum")
            elif pos_usd + q * bid_px > c.max_position_usd:
                notes.append("no bid: max position")
            else:
                bid = Target(bid_px, q, "add")
            if bid is not None and c.extra_levels > 0:   # never ladder past a touch level that itself
                running = [pos_usd + q * bid_px]           # didn't fit - that would leave a gap in front

                def add_q(i: int, px_i: Decimal, r=running) -> Optional[Decimal]:
                    qty_i = q_down(add_qty(long_, bid_imb_mult) * (c.level_size_mult ** i), m.step)
                    if qty_i <= 0:
                        return None
                    new_running = r[0] + qty_i * px_i
                    if abs(new_running) > c.max_position_usd:
                        return None                          # would breach the cap - stop laddering
                    r[0] = new_running
                    return qty_i
                extra_bids = ladder(BUY, bid_px, tb, bid_extra, "add", add_q)

        if ask_role == "reduce":
            q = reduce_qty()
            if sized_ok(q, ask_px):
                ask = Target(ask_px, q, "reduce")
            if c.extra_levels > 0:
                remaining = [abs(s.position) - (q if ask else ZERO)]

                def reduce_q(i: int, px_i: Decimal, r=remaining) -> Optional[Decimal]:
                    if r[0] <= 0:
                        return None
                    take = q_down(min(r[0], base * (c.level_size_mult ** i)), m.step)
                    if take <= 0:
                        return None
                    r[0] -= take
                    return take
                extra_asks = ladder(SELL, ask_px, ta, ask_extra, "reduce", reduce_q)
                start_px = extra_asks[-1].price if extra_asks else ask_px
                extra_asks = continue_adding(SELL, extra_asks, start_px, ta)
        else:
            why, code = blocked(SELL)
            blocked_map[SELL] = code
            q = add_qty(short_, ask_imb_mult)
            if why:
                notes.append(f"no ask: {why}")
            elif not sized_ok(q, ask_px):
                notes.append("no ask: size below minimum")
            elif pos_usd - q * ask_px < -c.max_position_usd:
                notes.append("no ask: max position")
            else:
                ask = Target(ask_px, q, "add")
            if ask is not None and c.extra_levels > 0:
                running = [pos_usd - q * ask_px]

                def add_q(i: int, px_i: Decimal, r=running) -> Optional[Decimal]:
                    qty_i = q_down(add_qty(short_, ask_imb_mult) * (c.level_size_mult ** i), m.step)
                    if qty_i <= 0:
                        return None
                    new_running = r[0] - qty_i * px_i
                    if abs(new_running) > c.max_position_usd:
                        return None
                    r[0] = new_running
                    return qty_i
                extra_asks = ladder(SELL, ask_px, ta, ask_extra, "add", add_q)
        return Plan(bid, ask, edge, skew, stress, notes, blocked_map, extra_bids, extra_asks)
