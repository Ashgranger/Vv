"""
Strategy S1 — Polymarket Bot  (clean rewrite)
==============================================

HOW IT WORKS
------------
1. SCAN: Poll orderbook every 2s per token.
   Setup = spread exactly 0.1¢ AND ask_wall < $10.

2. BUY: Place limit BUY at best bid.

3. TRACK FILL (WS + HTTP backup):
   - WebSocket fires order UPDATE → updates size_matched (actual tokens)
   - WebSocket fires trade CONFIRMED → triggers sell
   - HTTP polls every 30s as backup in case WS missed an event
   - On cancel (wall broke / bid dropped): HTTP check FIRST before cancelling

4. SELL: When fill confirmed → fetch live orderbook → sell at best ASK.
   Sell qty = size_matched (actual tokens in wallet), NOT trade.size (nominal).

5. CANCEL rules (after checking fill):
   - Ask wall ≥ $10 AND was < $10 at entry
   - Bid drops below entry bid

KEY FIX from previous versions:
   trade event `size` = nominal (can be > actual tokens received)
   order UPDATE `size_matched` = actual tokens in wallet → USE THIS for sell qty
   Selling nominal size caused "not enough balance" every time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aiohttp
import websockets
from dotenv import load_dotenv
from web3 import Web3

try:
    from web3.middleware import ExtraDataToPOAMiddleware as _POA
except ImportError:
    from web3.middleware import geth_poa_middleware as _POA  # type: ignore

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

# ─────────────────────────────────────────────────────────────────
# Logging — only important events
# ─────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("s1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

SEP = "═" * 66

# ─────────────────────────────────────────────────────────────────
# Config  (all tunable via .env)
# ─────────────────────────────────────────────────────────────────
class CFG:
    # Wallet / auth
    PRIVATE_KEY:  str = os.environ["PRIVATE_KEY"]
    PROXY_WALLET: str = os.environ["PROXY_WALLET"]
    CLOB_URL:     str = os.getenv("CLOB_HTTP_URL", "https://clob.polymarket.com").rstrip("/")
    RPC_URL:      str = os.getenv("RPC_URL", "https://polygon-rpc.com")
    USDC_ADDR:    str = os.getenv("USDC_CONTRACT_ADDRESS", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    SIG_TYPE:     int = int(os.getenv("SIG_TYPE", "0"))
    PM_API_KEY:   str = os.getenv("PM_API_KEY", "")
    PM_SECRET:    str = os.getenv("PM_API_SECRET", "")
    PM_PASS:      str = os.getenv("PM_PASSPHRASE", "")

    # Strategy S1 parameters
    SPREAD:       float = float(os.getenv("S1_SPREAD",    "0.1"))   # exact spread in cents
    ASK_WALL_MAX: float = float(os.getenv("S1_WALL_MAX",  "10"))    # ask wall must be < this
    BUY_SHARES:   float = float(os.getenv("S1_SHARES",    "5"))     # shares per buy order
    MIN_SELL:     float = float(os.getenv("S1_MIN_SELL",  "5"))     # min shares per sell
    PRICE_MIN:    float = float(os.getenv("S1_PRICE_MIN", "0.5"))   # min bid in cents
    PRICE_MAX:    float = float(os.getenv("S1_PRICE_MAX", "8.0"))   # max bid in cents
    POLL_SEC:     float = float(os.getenv("S1_POLL",      "2"))     # orderbook poll interval
    HB_SEC:       float = float(os.getenv("S1_HB",        "5"))     # heartbeat interval (< 10)
    FILL_POLL:    float = float(os.getenv("S1_FILL_POLL", "30"))    # HTTP fill backup interval

WS_URL       = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
BOOK_URL     = "https://clob.polymarket.com/book"
ORDER_URL    = "https://clob.polymarket.com/order"

USDC_ABI = [
    {"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf",
     "outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals",
     "outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
]

# ─────────────────────────────────────────────────────────────────
# Price helpers
# ─────────────────────────────────────────────────────────────────
def _p(level) -> float:
    return float(level["price"] if isinstance(level, dict) else level[0])

def _s(level) -> float:
    if isinstance(level, dict): return float(level.get("size", 0))
    return float(level[1]) if len(level) > 1 else 0.0

def cents(raw: float) -> float:
    """Raw CLOB price (0-1) → cent scale (0-100)."""
    return raw * 100 if raw <= 1.0 else raw

def raw(c: float) -> float:
    """Cent scale → raw CLOB price (0-1)."""
    return c / 100

def cost(cent_price: float, shares: float) -> float:
    return raw(cent_price) * shares

# ─────────────────────────────────────────────────────────────────
# Orderbook
# ─────────────────────────────────────────────────────────────────
@dataclass
class Book:
    bid: float; bid_sz: float
    ask: float; ask_sz: float
    spread: float

async def get_book(session: aiohttp.ClientSession, token_id: str) -> Optional[Book]:
    try:
        async with session.get(f"{BOOK_URL}?token_id={token_id}",
                               timeout=aiohttp.ClientTimeout(total=6)) as r:
            data = await r.json(content_type=None)
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return None
        bb = max(bids, key=_p);  ba = min(asks, key=_p)
        bid = round(cents(_p(bb)), 4);  bid_sz = round(_s(bb), 2)
        ask = round(cents(_p(ba)), 4);  ask_sz = round(_s(ba), 2)
        return Book(bid=bid, bid_sz=bid_sz, ask=ask, ask_sz=ask_sz,
                    spread=round(ask - bid, 4))
    except Exception as e:
        log.debug("book error %s: %s", token_id[-8:], e)
        return None

# ─────────────────────────────────────────────────────────────────
# Order state — one per tracked order
# ─────────────────────────────────────────────────────────────────
@dataclass
class OrderFill:
    """Tracks fill state for one open order. Updated by WS + HTTP backup."""
    order_id:     str
    original:     float          # shares placed
    matched:      float = 0.0    # actual tokens matched (from order UPDATE event)
    confirmed:    bool  = False  # True when trade CONFIRMED received on-chain
    cancelled:    bool  = False  # True when order CANCELLATION received
    failed:       bool  = False

# ─────────────────────────────────────────────────────────────────
# Fill Tracker — WS + HTTP backup
# ─────────────────────────────────────────────────────────────────
class FillTracker:
    """Single WS connection. Tracks all open orders. HTTP fallback every 30s."""

    def __init__(self, key: str, secret: str, passphrase: str) -> None:
        self._key    = key
        self._sec    = secret
        self._pass   = passphrase
        self._orders: dict[str, OrderFill] = {}
        self._lock   = asyncio.Lock()
        # Queue for fills on orders NOT tracked by strategy (safety-net seller)
        self.untracked_fills: asyncio.Queue = asyncio.Queue()

    async def track(self, order_id: str, original: float) -> OrderFill:
        async with self._lock:
            entry = OrderFill(order_id=order_id, original=original)
            self._orders[order_id] = entry
            return entry

    async def drop(self, order_id: str) -> None:
        async with self._lock:
            self._orders.pop(order_id, None)

    async def get(self, order_id: str) -> Optional[OrderFill]:
        async with self._lock:
            return self._orders.get(order_id)

    async def http_check(self, client: ClobClient, order_id: str) -> Optional[OrderFill]:
        """Poll one order via HTTP and update its OrderFill. Returns updated entry."""
        try:
            def _get():
                return client.get_order(order_id)
            resp = await asyncio.to_thread(_get)
            if resp is None:
                return None
            matched = float(resp.get("size_matched", 0) or 0) if isinstance(resp, dict) \
                      else float(getattr(resp, "size_matched", 0) or 0)
            status  = (resp.get("status", "") if isinstance(resp, dict)
                       else getattr(resp, "status", "")).upper()

            async with self._lock:
                entry = self._orders.get(order_id)
                if entry is None:
                    return None
                if matched > entry.matched:
                    log.info("📡 HTTP fill sync %s: matched %.4f→%.4f",
                             order_id[-12:], entry.matched, matched)
                    entry.matched = matched
                # NEVER treat MATCHED as confirmed — tokens not in wallet yet (balance=0)
                # Only TRADE CONFIRMED (from WS) or HTTP status=FILLED means tokens settled
                if status == "FILLED" and not entry.confirmed:
                    log.info("📡 HTTP: order %s FILLED → marking confirmed", order_id[-12:])
                    entry.confirmed = True
                return entry
        except Exception as e:
            log.debug("http_check %s: %s", order_id[-12:], e)
            return None

    # ── Background WS ────────────────────────────────────────────

    async def run(self) -> None:
        while True:
            try:
                await self._ws_connect()
            except Exception as e:
                log.warning("WS error: %s — reconnect in 3s", e)
                await asyncio.sleep(3)

    async def _ws_connect(self) -> None:
        log.info("WS: connecting")
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            log.info("WS: ✅ connected")
            await ws.send(json.dumps({
                "type": "user",
                "auth": {"apiKey": self._key, "secret": self._sec, "passphrase": self._pass}
            }))
            log.info("WS: 📡 subscribed")
            ping = asyncio.create_task(self._ping(ws))
            try:
                async for msg in ws:
                    if not msg or msg in ("PONG", "PING"):
                        continue
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    await self._dispatch(data)
            finally:
                ping.cancel()

    async def _ping(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(10)
                await ws.send("PING")
        except asyncio.CancelledError:
            pass

    async def _dispatch(self, data: dict) -> None:
        etype = data.get("event_type")
        if etype == "order":
            await self._on_order(data)
        elif etype == "trade":
            await self._on_trade(data)

    async def _on_order(self, o: dict) -> None:
        oid       = o.get("id", "")
        otype     = o.get("type", "")
        matched   = float(o.get("size_matched", 0) or 0)
        asset_id  = o.get("asset_id", "")
        outcome   = o.get("outcome", "")
        side      = o.get("side", "")

        async with self._lock:
            entry = self._orders.get(oid)
            if entry is None:
                # Untracked order — push to safety-net queue if it has a fill
                if otype == "UPDATE" and matched > 0 and side == "BUY":
                    log.info("🔔 UNTRACKED fill: order %s matched=%.4f asset=%s",
                             oid[-12:], matched, asset_id[-12:])
                    await self.untracked_fills.put({
                        "order_id": oid,
                        "token_id": asset_id,
                        "matched":  matched,
                        "source":   "order_update",
                    })
                return
            if otype == "UPDATE":
                entry.matched = matched
                log.info("WS: ORDER UPDATE %s | matched=%.4f", oid[-12:], matched)
            elif otype == "CANCELLATION":
                entry.cancelled = True
                log.info("WS: ORDER CANCELLED %s | matched=%.4f", oid[-12:], matched)

    async def _on_trade(self, t: dict) -> None:
        # Trade events reference the taker order. Also check maker_orders.
        taker  = t.get("taker_order_id", "")
        status = t.get("status", "").upper()
        size   = float(t.get("size", 0) or 0)
        price  = float(t.get("price", 0) or 0)

        makers = [m.get("order_id", "") for m in (t.get("maker_orders") or [])]
        candidates = [taker] + makers

        async with self._lock:
            for oid in candidates:
                entry = self._orders.get(oid)
                if entry is None:
                    continue
                if status == "CONFIRMED":
                    entry.confirmed = True
                    log.info("WS: TRADE CONFIRMED %s | size=%.4f @ %.5f",
                             oid[-12:], size, price)
                elif status == "MATCHED":
                    log.info("WS: TRADE MATCHED %s | size=%.4f @ %.5f (pending)",
                             oid[-12:], size, price)
                elif status == "MINED":
                    log.info("WS: TRADE MINED %s", oid[-12:])
                elif status == "FAILED":
                    entry.failed = True
                    log.warning("WS: TRADE FAILED %s", oid[-12:])

        # Also handle CONFIRMED for untracked orders (previous-session fills etc)
        if status == "CONFIRMED":
            asset_id  = t.get("asset_id", "")
            tside     = t.get("trader_side", "")  # TAKER or MAKER
            trade_side = t.get("side", "")        # BUY or SELL
            async with self._lock:
                all_tracked = set(self._orders.keys())
            untracked_ids = [c for c in candidates if c not in all_tracked]
            for uid in untracked_ids:
                if uid and asset_id and trade_side == "SELL":
                    # We were the SELLER — no action needed
                    continue
                if uid and asset_id:
                    log.info("🔔 UNTRACKED CONFIRMED: order %s size=%.4f token=%s",
                             uid[-12:], size, asset_id[-12:])
                    await self.untracked_fills.put({
                        "order_id": uid,
                        "token_id": asset_id,
                        "matched":  size,
                        "source":   "trade_confirmed",
                    })

# ─────────────────────────────────────────────────────────────────
# Budget
# ─────────────────────────────────────────────────────────────────
class Budget:
    def __init__(self, balance: float) -> None:
        self._total = balance
        self._avail = balance
        self._open  = 0
        self._lock  = asyncio.Lock()

    async def reserve(self, amount: float) -> bool:
        async with self._lock:
            if self._avail < amount - 1e-9:
                log.debug("budget: skip $%.5f (avail $%.5f)", amount, self._avail)
                return False
            self._avail -= amount
            self._open  += 1
            log.info("💰 reserved $%.5f | avail $%.5f | orders=%d",
                     amount, self._avail, self._open)
            return True

    async def release(self, amount: float, reason: str = "") -> None:
        async with self._lock:
            self._avail  = min(self._total, self._avail + amount)
            self._open   = max(0, self._open - 1)
            log.info("💰 released $%.5f (%s) | avail $%.5f | orders=%d",
                     amount, reason, self._avail, self._open)

# ─────────────────────────────────────────────────────────────────
# CLOB wrappers (sync → async via thread)
# ─────────────────────────────────────────────────────────────────
def _buy_sync(client, token_id, cent_price, shares):
    try:
        signed = client.create_order(
            OrderArgs(token_id=token_id, price=raw(cent_price), size=shares, side="BUY"))
        resp = client.post_order(signed, OrderType.GTC)
        ok  = getattr(resp, "success", False) or (isinstance(resp, dict) and resp.get("success"))
        oid = getattr(resp, "orderID", None) or (resp.get("orderID") if isinstance(resp, dict) else None)
        if ok and oid:
            return oid
        err = resp.get("error", resp.get("errorMsg", str(resp))) if isinstance(resp, dict) else str(resp)
        log.error("BUY rejected: %s", err)
    except Exception as e:
        log.error("BUY exception: %s", e)
    return None

def _sell_sync(client, token_id, cent_price, shares):
    """Try sell. If balance error, retry with shares - 0.05."""
    for qty in [shares, shares - 0.05]:
        if qty <= 0:
            break
        try:
            signed = client.create_order(
                OrderArgs(token_id=token_id, price=raw(cent_price), size=qty, side="SELL"))
            resp = client.post_order(signed, OrderType.GTC)
            ok  = getattr(resp, "success", False) or (isinstance(resp, dict) and resp.get("success"))
            oid = getattr(resp, "orderID", None) or (resp.get("orderID") if isinstance(resp, dict) else None)
            if ok and oid:
                return oid
            err = resp.get("error", resp.get("errorMsg", str(resp))) if isinstance(resp, dict) else str(resp)
            if "balance" in str(err).lower() and qty == shares:
                log.warning("SELL balance error — retry with %.4f", qty - 0.05)
                continue
            log.error("SELL rejected: %s", err)
            return None
        except Exception as e:
            log.error("SELL exception: %s", e)
            return None
    return None

def _cancel_sync(client, order_id):
    try:
        client.cancel_order(order_id)
        return True
    except Exception as e:
        log.debug("cancel %s: %s", order_id[-12:], e)
        return False

async def do_buy(client, token_id, cent_price, shares):
    return await asyncio.to_thread(_buy_sync, client, token_id, cent_price, shares)

async def do_sell(client, token_id, cent_price, shares):
    if shares < CFG.MIN_SELL:
        log.info("sell %.4f < min %.0f — skip", shares, CFG.MIN_SELL)
        return None
    log.info("  🔖 SELL %.4f @ %.3f¢ (raw=%.5f)", shares, cent_price, raw(cent_price))
    oid = await asyncio.to_thread(_sell_sync, client, token_id, cent_price, shares)
    if oid:
        log.info("  ✅ SELL order placed | id=%s", oid)
    else:
        log.error("  ❌ SELL FAILED for %.4f shares @ %.3f¢", shares, cent_price)
    return oid

async def do_cancel(client, order_id):
    return await asyncio.to_thread(_cancel_sync, client, order_id)

# ─────────────────────────────────────────────────────────────────
# Heartbeat (prevents CLOB auto-cancelling open orders after 10s)
# ─────────────────────────────────────────────────────────────────
async def heartbeat_loop(client: ClobClient) -> None:
    log.info("💓 heartbeat started (%.0fs)", CFG.HB_SEC)
    hb_id = ""
    while True:
        await asyncio.sleep(CFG.HB_SEC)
        try:
            def _hb():
                return client.post_heartbeat(hb_id)
            resp  = await asyncio.to_thread(_hb)
            hb_id = (resp.get("heartbeat_id", "") if isinstance(resp, dict)
                     else getattr(resp, "heartbeat_id", ""))
        except Exception as e:
            err = str(e)
            if any(x in err for x in ("425", "Invalid Heartbeat", "Request exception")):
                pass   # transient — ignore
            else:
                log.warning("💓 heartbeat: %s", e)
            hb_id = ""

# ─────────────────────────────────────────────────────────────────
# Token state
# ─────────────────────────────────────────────────────────────────
class S(Enum):
    WATCHING = "WATCHING"
    PENDING  = "PENDING"   # order placed, waiting for fill

@dataclass
class TS:
    token_id:   str
    token_type: str
    question:   str
    state:       S      = S.WATCHING
    order_id:    str    = ""
    entry_bid:   float  = 0.0
    entry_ask_sz:float  = 0.0    # ask wall at placement
    reserved:    float  = 0.0    # USDC locked
    original:    float  = 0.0    # shares placed
    sold:        float  = 0.0    # shares we've placed sell orders for
    last_act:    float  = 0.0    # last matched amount we acted on
    last_poll:   float  = 0.0    # last HTTP fill poll (monotonic)
    cancels:     int    = 0      # consecutive cancels
    skip_until:  float  = 0.0    # skip setups until this monotonic time

# ─────────────────────────────────────────────────────────────────
# Core sell logic — used both on fill and on cancel-with-fill
# ─────────────────────────────────────────────────────────────────
async def sell_filled(
    session: aiohttp.ClientSession,
    client:  ClobClient,
    budget:  Budget,
    tracker: FillTracker,
    ts:      TS,
    entry:   OrderFill,
    label:   str = "",
) -> None:
    """Sell whatever has been matched. Always fetches live ask price."""
    to_sell = round(entry.matched - ts.sold, 4)
    if to_sell < CFG.MIN_SELL:
        log.info("[%s] ⏳ %.4f confirmed < min %.0f — waiting for more fill",
                 label, to_sell, CFG.MIN_SELL)
        # Advance last_act so we don't re-trigger every poll for the same amount
        ts.last_act = entry.matched
        return

    # Fetch live book for current ask
    book = await get_book(session, ts.token_id)
    ask  = book.ask if book else ts.entry_bid + CFG.SPREAD  # fallback

    log.info("[%s] 📤 matched=%.4f sold=%.4f → SELL %.4f @ ask=%.3f¢",
             label, entry.matched, ts.sold, to_sell, ask)

    oid = await do_sell(client, ts.token_id, ask, to_sell)
    if oid:
        ts.sold    += to_sell
        ts.last_act = entry.matched
        fill_cost   = cost(ts.entry_bid, to_sell)
        await budget.release(fill_cost, f"sold ({label})")
        ts.reserved = max(0.0, ts.reserved - fill_cost)

# ─────────────────────────────────────────────────────────────────
# Per-token monitor
# ─────────────────────────────────────────────────────────────────
async def monitor(
    session: aiohttp.ClientSession,
    client:  ClobClient,
    budget:  Budget,
    tracker: FillTracker,
    ts:      TS,
) -> None:
    lbl = f"{ts.token_type}…{ts.token_id[-8:]}"

    while True:
        await asyncio.sleep(CFG.POLL_SEC)
        try:
            await _tick(session, client, budget, tracker, ts, lbl)
        except Exception as e:
            import traceback
            log.error("[%s] tick error: %s\n%s", lbl, e, traceback.format_exc())

async def _tick(session, client, budget, tracker, ts, lbl):

    # ── WATCHING ────────────────────────────────────────────────
    if ts.state == S.WATCHING:

        # Per-token cooldown
        if ts.skip_until > 0 and time.monotonic() < ts.skip_until:
            return

        book = await get_book(session, ts.token_id)
        if book is None:
            return

        # Exact spread check (±0.005 tolerance for float imprecision)
        if abs(book.spread - CFG.SPREAD) >= 0.005:
            return
        # Ask wall must be < max
        if book.ask_sz >= CFG.ASK_WALL_MAX:
            return
        # Price range filter
        if not (CFG.PRICE_MIN <= book.bid <= CFG.PRICE_MAX):
            return

        # Budget
        c = cost(book.bid, CFG.BUY_SHARES)
        if not await budget.reserve(c):
            return

        log.info("[%s] ✅ SETUP bid=%.3f¢($%.1f) ask=%.3f¢($%.1f) spread=%.3f cost=$%.5f",
                 lbl, book.bid, book.bid_sz, book.ask, book.ask_sz, book.spread, c)

        oid = await do_buy(client, ts.token_id, book.bid, CFG.BUY_SHARES)
        if not oid:
            await budget.release(c, "buy failed")
            ts.cancels += 1
            return

        entry = await tracker.track(oid, CFG.BUY_SHARES)

        ts.state        = S.PENDING
        ts.order_id     = oid
        ts.entry_bid    = book.bid
        ts.entry_ask_sz = book.ask_sz
        ts.reserved     = c
        ts.original     = CFG.BUY_SHARES
        ts.sold         = 0.0
        ts.last_act     = 0.0
        ts.last_poll    = time.monotonic()

        log.info("[%s] 📌 ORDER PLACED | id=%s | bid=%.3f¢ | ask_wall=$%.1f",
                 lbl, oid, book.bid, book.ask_sz)

    # ── PENDING ──────────────────────────────────────────────────
    elif ts.state == S.PENDING:

        entry = await tracker.get(ts.order_id)
        if entry is None:
            # Lost track — go back to watching
            log.warning("[%s] lost OrderFill — reset", lbl)
            await budget.release(ts.reserved, "lost entry")
            ts.state = S.WATCHING
            return

        # ── HTTP backup poll every FILL_POLL seconds ─────────────
        now = time.monotonic()
        if now - ts.last_poll >= CFG.FILL_POLL:
            ts.last_poll = now
            await tracker.http_check(client, ts.order_id)
            # Re-fetch entry after update
            entry = await tracker.get(ts.order_id)
            if entry is None:
                await budget.release(ts.reserved, "lost after poll")
                ts.state = S.WATCHING
                return

        # ── FILL CHECK (always first) ────────────────────────────
        #
        # Sell trigger: trade CONFIRMED on-chain
        # Sell qty:     entry.matched (size_matched from order UPDATE)
        #               NOT trade.size which is nominal
        #
        if entry.confirmed and entry.matched > ts.last_act:
            log.info("[%s] 🎉 CONFIRMED fill: matched=%.4f confirmed=True",
                     lbl, entry.matched)
            await sell_filled(session, client, budget, tracker, ts, entry, lbl)

            is_full = entry.matched >= ts.original * 0.99
            if is_full:
                await tracker.drop(ts.order_id)
                await budget.release(ts.reserved, "fully filled")
                ts.state = S.WATCHING
                return

        # ── CANCEL CONDITIONS (after fill check) ─────────────────
        book = await get_book(session, ts.token_id)
        if book is None:
            return

        should_cancel = False
        cancel_reason = ""

        # Ask wall broke (only if it was small at entry)
        if ts.entry_ask_sz < CFG.ASK_WALL_MAX and book.ask_sz >= CFG.ASK_WALL_MAX:
            should_cancel = True
            cancel_reason = f"ask_wall ${book.ask_sz:.1f}≥${CFG.ASK_WALL_MAX:.0f}"

        # Bid dropped below entry
        elif book.bid < ts.entry_bid:
            should_cancel = True
            cancel_reason = f"bid {book.bid:.3f} < entry {ts.entry_bid:.3f}"

        if should_cancel:
            # HTTP check BEFORE cancelling — order may have filled while we weren't watching
            log.info("[%s] ⚠️  %s — HTTP check before cancel", lbl, cancel_reason)
            fresh = await tracker.http_check(client, ts.order_id)
            if fresh and fresh.matched >= CFG.MIN_SELL and fresh.matched > ts.sold:
                log.info("[%s] Fill found before cancel! matched=%.4f — selling first", lbl, fresh.matched)
                await sell_filled(session, client, budget, tracker, ts, fresh, lbl)

            # Cancel the order
            await do_cancel(client, ts.order_id)
            await tracker.drop(ts.order_id)
            # Release only the unfilled portion of budget
            unfilled_cost = cost(ts.entry_bid, max(0.0, ts.original - ts.sold))
            await budget.release(unfilled_cost, cancel_reason)
            ts.reserved = 0.0

            ts.cancels += 1
            # Exponential backoff: 30s, 60s, 120s, 240s, max 600s
            backoff = min(600.0, 30.0 * (2 ** min(ts.cancels - 1, 4)))
            ts.skip_until = time.monotonic() + backoff
            ts.state = S.WATCHING

            log.info("[%s] 🗑️  CANCELLED (%s) | ⏸️ skip %.0fs (cancel #%d)",
                     lbl, cancel_reason, backoff, ts.cancels)

        # Trade failed
        elif entry.failed:
            log.warning("[%s] trade FAILED — reset", lbl)
            await do_cancel(client, ts.order_id)
            await tracker.drop(ts.order_id)
            await budget.release(ts.reserved, "trade failed")
            ts.state = S.WATCHING

# ─────────────────────────────────────────────────────────────────
# CLOB client factory
# ─────────────────────────────────────────────────────────────────
def make_client() -> ClobClient:
    funder = CFG.PROXY_WALLET if CFG.SIG_TYPE in (1, 2) else None
    log.info("SigType=%d | Wallet=%s", CFG.SIG_TYPE, CFG.PROXY_WALLET)

    anon = ClobClient(host=CFG.CLOB_URL, chain_id=POLYGON,
                      key=CFG.PRIVATE_KEY, signature_type=CFG.SIG_TYPE, funder=funder)
    creds = None
    for method in ("create_api_key", "derive_api_key"):
        try:
            creds = getattr(anon, method)()
            if creds:
                log.info("API creds via %s ✅", method)
                break
        except Exception as e:
            log.debug("%s: %s", method, e)

    if not creds:
        raise RuntimeError("Cannot get API credentials — connect wallet on polymarket.com first")

    return ClobClient(host=CFG.CLOB_URL, chain_id=POLYGON,
                      key=CFG.PRIVATE_KEY, creds=creds,
                      signature_type=CFG.SIG_TYPE, funder=funder)

def fetch_balance(address: str) -> float:
    w3   = Web3(Web3.HTTPProvider(CFG.RPC_URL))
    w3.middleware_onion.inject(_POA, layer=0)
    usdc = w3.eth.contract(address=Web3.to_checksum_address(CFG.USDC_ADDR), abi=USDC_ABI)
    raw_ = usdc.functions.balanceOf(Web3.to_checksum_address(address)).call()
    dec  = usdc.functions.decimals().call()
    return raw_ / (10 ** dec)

# ─────────────────────────────────────────────────────────────────
# Safety-net seller
# Catches fills for ANY order — including those from previous sessions,
# orders placed manually, or fills the strategy loop missed.
# Triggered by WS events; sells at live best ask.
# ─────────────────────────────────────────────────────────────────
async def safety_net_seller(
    session: aiohttp.ClientSession,
    client:  ClobClient,
    tracker: FillTracker,
) -> None:
    log.info("🛡️  Safety-net seller active")
    # Track which (order_id, token_id) pairs we've already sold to avoid duplicates
    handled: set[str] = set()

    while True:
        try:
            item = await asyncio.wait_for(tracker.untracked_fills.get(), timeout=30)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            log.debug("safety_net queue error: %s", e)
            continue

        order_id = item.get("order_id", "")
        token_id = item.get("token_id", "")
        matched  = float(item.get("matched", 0))
        source   = item.get("source", "")

        key = f"{order_id}:{round(matched, 4)}"
        if key in handled:
            continue
        handled.add(key)

        if not token_id or matched < CFG.MIN_SELL:
            log.info("🛡️  skip %s: token=%s matched=%.4f < min %.0f",
                     order_id[-12:], token_id[-12:] if token_id else "?", matched, CFG.MIN_SELL)
            continue

        log.info("🛡️  UNTRACKED FILL | order=%s token=%s matched=%.4f (source=%s)",
                 order_id[-12:], token_id[-12:], matched, source)

        # Fetch live book to get best ask
        book = await get_book(session, token_id)
        if book is None:
            log.error("🛡️  no orderbook for token %s — cannot sell", token_id[-12:])
            continue

        log.info("🛡️  SELL %.4f @ ask=%.3f¢ (live)", matched, book.ask)
        oid = await do_sell(client, token_id, book.ask, matched)
        if oid:
            log.info("🛡️  ✅ SELL placed | id=%s", oid)
        else:
            log.error("🛡️  ❌ SELL failed for %.4f @ %.3f¢", matched, book.ask)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
async def main() -> None:
    log.info(SEP)
    log.info("🚀  Strategy S1")
    log.info("  spread=%.2f¢ | wall<$%.0f | buy=%.0f sh | min_sell=%.0f sh "
             "| price=[%.1f-%.1f]¢ | poll=%.1fs",
             CFG.SPREAD, CFG.ASK_WALL_MAX, CFG.BUY_SHARES, CFG.MIN_SELL,
             CFG.PRICE_MIN, CFG.PRICE_MAX, CFG.POLL_SEC)
    log.info(SEP)

    # Balance
    try:
        bal = fetch_balance(CFG.PROXY_WALLET)
        log.info("💵 USDC balance: $%.5f", bal)
    except Exception as e:
        log.error("Balance fetch failed: %s", e)
        return
    if bal <= 0:
        log.error("Zero balance — deposit USDC.e on Polygon")
        return

    # Load markets
    try:
        with open("filtered_markets.json") as f:
            markets = json.load(f)
    except FileNotFoundError:
        log.error("filtered_markets.json not found — run market_scanner.py first")
        return

    tokens = []
    for m in markets:
        if m.get("yes_valid"):
            tokens.append(TS(token_id=m["yes_token"], token_type="YES", question=m["question"]))
        if m.get("no_valid"):
            tokens.append(TS(token_id=m["no_token"], token_type="NO", question=m["question"]))

    if not tokens:
        log.error("No eligible tokens — re-run market_scanner.py")
        return

    log.info("🎯 Monitoring %d tokens", len(tokens))
    log.info(SEP)

    # CLOB client
    try:
        client = make_client()
        log.info("✅ ClobClient ready")
    except Exception as e:
        log.error("CLOB setup failed: %s", e)
        return

    # Cancel any stale open orders from previous runs
    try:
        log.info("🧹 cancelling stale orders from previous runs...")
        def _cancel_all():
            return client.cancel_all()
        resp = await asyncio.to_thread(_cancel_all)
        log.info("🧹 cancel_all: %s", resp)
    except Exception as e:
        log.warning("cancel_all failed: %s", e)

    # WS credentials
    ws_key  = CFG.PM_API_KEY  or ""
    ws_sec  = CFG.PM_SECRET   or ""
    ws_pass = CFG.PM_PASS     or ""
    if not all([ws_key, ws_sec, ws_pass]):
        log.error("WS credentials missing — add PM_API_KEY / PM_API_SECRET / PM_PASSPHRASE to .env")
        return

    tracker = FillTracker(ws_key, ws_sec, ws_pass)
    budget  = Budget(bal)

    connector = aiohttp.TCPConnector(limit=80)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            tracker.run(),
            heartbeat_loop(client),
            safety_net_seller(session, client, tracker),
            *[monitor(session, client, budget, tracker, ts) for ts in tokens],
        )

if __name__ == "__main__":
    asyncio.run(main())
