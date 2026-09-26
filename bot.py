"""Level 7 Market Maker for Arcus Perpetuals."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from decimal import Decimal
from typing import Any, Optional

from config import Config
from exchange import Exchange
from market import Market, MarketData
from signer import Signer
from ledger import Ledger, Fill
from engine import MarketMakingEngine, QuoteTarget
from orders import OrderManager, Order
from utils import BPS, BUY, SELL, ZERO, ONE, Fatal, fmt

log = logging.getLogger("bot")


def extract_positions(c: Any) -> list:
    if isinstance(c, list):
        return [r for r in c if isinstance(r, dict)]
    if isinstance(c, dict):
        if "positions" in c:
            p = c["positions"]
            if isinstance(p, dict):
                return [r for r in p.values() if isinstance(r, dict)]
            if isinstance(p, list):
                return [r for r in p if isinstance(r, dict)]
            return []
        if "marketId" in c:
            return [c]
    return []


class MarketMaker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.now = time.monotonic
        self.stop_evt = asyncio.Event()

        self.ex = Exchange(cfg, self._on_channel)
        self.md = MarketData(cfg)
        self.signer = Signer(cfg.signing_key, cfg.address, cfg.account_index)
        self.ledger = Ledger(cfg)
        self.engine = MarketMakingEngine(cfg)
        self.om = OrderManager(cfg, self.ex, self.signer, self._get_market, self._on_fill)

        self._recent_fills: deque = deque()
        self._burst_blocked_until = {BUY: 0.0, SELL: 0.0}
        self._trend_blocked_until = {BUY: 0.0, SELL: 0.0}

        self._last_heartbeat = 0.0
        self._last_reconcile = 0.0
        self._last_status = 0.0
        self._last_info_fetch = 0.0
        self._tick_lock = asyncio.Lock()
        self._dirty_evt = asyncio.Event()

    def _get_market(self) -> Market:
        if not self.md.info:
            raise Fatal("Market metadata not yet loaded")
        return self.md.info

    def _on_channel(self, channel: str, contents: Any, is_snapshot: bool) -> None:
        now = self.now()
        if channel == "bbo":
            if not isinstance(contents, dict):
                return
            bb, ba = contents.get("bestBid"), contents.get("bestAsk")
            if not bb or not ba:
                return
            try:
                bid = Decimal(str(bb["price"]))
                ask = Decimal(str(ba["price"]))
                bid_sz = Decimal(str(bb["size"])) if "size" in bb else None
                ask_sz = Decimal(str(ba["size"])) if "size" in ba else None
                self.md.update(bid, ask, bid_sz, ask_sz, now)
                if self.md.mid:
                    self.ledger.process_markouts(self.md.mid, now)
                self._dirty_evt.set()
            except Exception:
                pass

        elif channel == "trades":
            if isinstance(contents, list):
                for tr in contents:
                    self._handle_trade(tr, now)
            elif isinstance(contents, dict):
                self._handle_trade(contents, now)

        elif channel == "orderBook":
            if isinstance(contents, dict):
                bids = contents.get("bids") or []
                asks = contents.get("asks") or []
                self.md.on_depth(bids, asks, now)

        elif channel == "orders":
            if isinstance(contents, list):
                for row in contents:
                    self.om.on_update(row, now)
            elif isinstance(contents, dict):
                self.om.on_update(contents, now)
            self._dirty_evt.set()

        elif channel == "positions":
            rows = extract_positions(contents)
            m = self.md.info
            if m:
                target_mid = self.md.mid or m.mark
                for r in rows:
                    if int(r.get("marketId", -1)) == m.market_id:
                        side = str(r.get("side", "FLAT")).upper()
                        sz = Decimal(str(r.get("size", "0")))
                        signed_pos = sz if side == "LONG" else (-sz if side == "SHORT" else ZERO)
                        self.ledger.reconcile(signed_pos, now, target_mid, m.min_notional)

    def _handle_trade(self, tr: dict, now: float) -> None:
        try:
            side = str(tr.get("side") or tr.get("orderSide") or "BUY").upper()
            sz = Decimal(str(tr.get("size") or tr.get("quantity") or "0"))
            px = Decimal(str(tr.get("price") or "0"))
            if sz > 0:
                self.md.on_trade(side, sz, px, now)
        except Exception:
            pass

    def _on_fill(self, side: str, qty: Decimal, price: Decimal, o: Order) -> None:
        now = self.now()
        m = self.md.info
        mid = self.md.mid or price
        min_notional = m.min_notional if m else Decimal("5")
        
        fill = self.ledger.on_fill(side, qty, price, mid, now, min_notional)
        log.info("FILL L%d %s %s @ %s | edge=%sbps pos=%s pnl=$%s",
                 o.pair_index, side, fmt(qty), fmt(price), fmt(fill.edge_bps),
                 fmt(self.ledger.position), fmt(self.ledger.total_pnl(mid)))

        self._recent_fills.append((now, side))
        while self._recent_fills and now - self._recent_fills[0][0] > self.cfg.burst_window_s:
            self._recent_fills.popleft()

        same_side = sum(1 for _, s in self._recent_fills if s == side)
        if same_side >= self.cfg.burst_fills:
            self._burst_blocked_until[side] = now + self.cfg.burst_cooldown_s
            log.warning("BURST GUARD: %d %s fills in %.1fs -> pulling %s for %.1fs",
                        same_side, side, self.cfg.burst_window_s, side, self.cfg.burst_cooldown_s)
            asyncio.create_task(self.om.cancel_side(side, now))

        rapid_fills = sum(1 for t, s in self._recent_fills if s == side and (now - t) <= self.cfg.sweep_guard_window_s)
        if rapid_fills >= self.cfg.sweep_guard_fills:
            self._burst_blocked_until[side] = max(self._burst_blocked_until[side], now + self.cfg.burst_cooldown_s)
            log.warning("SWEEP GUARD: %d %s fills in <=%.1fs -> emergency cancel %s",
                        rapid_fills, side, self.cfg.sweep_guard_window_s, side)
            asyncio.create_task(self.om.cancel_side(side, now))

        self._journal(fill)
        self._dirty_evt.set()

    def _journal(self, f: Fill) -> None:
        if not self.cfg.journal_path or self.cfg.journal_path == os.devnull:
            return
        row = {
            "ts": f.ts, "side": f.side, "qty": fmt(f.qty), "price": fmt(f.price),
            "mid": fmt(f.mid), "edge_bps": fmt(f.edge_bps), "pos": fmt(f.position),
            "realized_delta": fmt(f.realized_delta), "total_realized": fmt(self.ledger.realized),
            "fees": fmt(self.ledger.fees)
        }
        try:
            with open(self.cfg.journal_path, "a") as fp:
                fp.write(json.dumps(row) + "\n")
        except Exception:
            pass

    async def tick(self) -> None:
        async with self._tick_lock:
            now = self.now()
            m = self.md.info
            if not m:
                return

            mid = self.md.mid
            if not mid or not self.md.bid or not self.md.ask:
                return

            tot_pnl = self.ledger.total_pnl(mid)
            if tot_pnl <= -self.cfg.session_max_loss_usd:
                log.error("SESSION MAX LOSS BREACHED ($%s <= -$%s) - HALTING",
                          fmt(tot_pnl), fmt(self.cfg.session_max_loss_usd))
                await self.om.cancel_all()
                self.stop_evt.set()
                return

            if self.md.spread_bps > self.cfg.max_market_spread_bps:
                await self.om.cancel_all()
                return

            if m.mark and abs(mid - m.mark) / m.mark * BPS > self.cfg.max_oracle_dev_bps:
                await self.om.cancel_all()
                return

            if m.is_outside_rth and not self.cfg.quote_outside_rth:
                await self.om.cancel_all()
                return

            if self.md.jump_active(now):
                await self.om.cancel_all()
                return

            buy_blocked = (now < self._burst_blocked_until[BUY] or now < self._trend_blocked_until[BUY])
            sell_blocked = (now < self._burst_blocked_until[SELL] or now < self._trend_blocked_until[SELL])

            ret_trend = self.md.ret_bps(self.cfg.trend_window_s, now)
            if ret_trend <= -self.cfg.trend_pull_bps:
                self._trend_blocked_until[BUY] = now + self.cfg.trend_hold_s
                buy_blocked = True
            elif ret_trend >= self.cfg.trend_pull_bps:
                self._trend_blocked_until[SELL] = now + self.cfg.trend_hold_s
                sell_blocked = True

            pos_usd = self.ledger.position * mid
            if self.md.move_bps(self.cfg.vol_window_s, now) >= self.cfg.vol_pause_bps:
                # Volatility spike: pause ADDING sides, never pause UNWIND sides
                if pos_usd >= 0:
                    buy_blocked = True
                if pos_usd <= 0:
                    sell_blocked = True

            existing_slots = set(self.om.pair_slots.keys())
            targets = self.engine.generate_ladder_quotes(
                m, self.md, self.ledger, now, buy_blocked, sell_blocked, existing_slots=existing_slots
            )

            await self.om.sync_quotes(targets, now)

    async def _heartbeat(self, now: float) -> None:
        if now - self._last_heartbeat < self.cfg.heartbeat_s:
            return
        self._last_heartbeat = now
        try:
            await self.ex.call("post", {"type": "heartbeat", "payload": {}}, timeout=4.0)
        except Exception:
            pass

    async def _reconcile(self, now: float) -> None:
        if now - self._last_reconcile < self.cfg.reconcile_s:
            return
        self._last_reconcile = now
        try:
            m = self.md.info
            if not m:
                return
            res = await self.ex.get("orders", {"address": self.cfg.address, "accountIndex": self.cfg.account_index,
                                                "marketId": m.market_id})
            if res and "openOrders" in res:
                await self.om.reconcile(res["openOrders"], now)
        except Exception:
            pass

    def _status_log(self, now: float) -> None:
        if now - self._last_status < self.cfg.status_s:
            return
        self._last_status = now
        mid = self.md.mid or Decimal("0")
        regime = self.md.detect_regime(now, self.ledger.tox_bps)
        log.info("STATUS | %s | mid=%s spr=%sbps obi=%s vol=%sbps | pos=%s unreal=$%s pnl=$%s | orders: %s",
                 regime, fmt(mid), fmt(self.md.spread_bps), fmt(self.md.obi), fmt(self.md.vol_bps),
                 fmt(self.ledger.position), fmt(self.ledger.unrealized(mid)),
                 fmt(self.ledger.total_pnl(mid)), self.om.describe(now))

    async def run(self) -> None:
        log.info("Connecting to %s Arcus WS (%s)...", self.cfg.env_name, self.ex.ws_url)
        raw_markets = await self.ex.fetch_markets(self.cfg.market)
        self.md.info = Market.from_api(raw_markets[0])
        self.md.info_ts = self.now()
        log.info("Market loaded: %s (ID %d) tick=%s step=%s min_notional=$%s",
                 self.md.info.name, self.md.info.market_id, self.md.info.tick,
                 self.md.info.step, self.md.info.min_notional)

        try:
            import websockets
        except ImportError:
            log.error("websockets package not available; install via pip install websockets")
            return

        async with websockets.connect(self.ex.ws_url, ping_interval=15, max_size=2**23) as ws:
            self.ex.ws = ws
            reader_task = asyncio.create_task(self.ex.reader())

            await self.ex.subscribe("bbo", self.cfg.market)
            await self.ex.subscribe("orderBook", self.cfg.market)
            await self.ex.subscribe("trades", self.cfg.market)
            await self.ex.subscribe("orders", self.cfg.address)
            await self.ex.subscribe("userFills", self.cfg.address)
            await self.ex.subscribe("positions", self.cfg.address)

            if self.om.maybe_orders:
                await self.om.cancel_all()

            log.info("Subscribed to data feeds. Level 7 MM Engine active.")

            while not self.stop_evt.is_set():
                now = self.now()
                await self._heartbeat(now)
                await self._reconcile(now)
                self._status_log(now)

                await self.tick()

                try:
                    await asyncio.wait_for(self._dirty_evt.wait(), timeout=self.cfg.loop_s)
                    self._dirty_evt.clear()
                except asyncio.TimeoutError:
                    pass

            log.info("Stopping bot - cancelling all resting orders...")
            await self.om.cancel_all()
            reader_task.cancel()
