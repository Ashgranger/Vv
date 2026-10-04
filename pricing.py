"""Quantitative Pricing Engine.

Implements:
1. Micro-Price & Imbalance Fair Value S*
2. Avellaneda-Stoikov Reservation Price r(s, q) with non-linear aversion
3. Dynamic Half-Spread Scaling across 1-8 bps regimes
4. Asymmetric Quoting & Inventory Unwinding
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from config import Config
from models import MarketSnapshot, q_down, q_up, clamp


class PricingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_fair_value(self, state: MarketSnapshot) -> float:
        """Estimates high-frequency fair value S* using micro-price, OBI, and TFI."""
        mid = state.mid
        if not self.cfg.enable_micro_intel:
            return mid

        micro = state.micro_price
        obi = state.obi
        half_spread = (state.best_ask - state.best_bid) / 2.0

        # Calculate Trade Flow Imbalance (TFI) over recent 2.5s window
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

    def compute_reservation_price(self, fair_value: float, position: float, vol_bps: float) -> float:
        """Computes inventory-adjusted reservation price r(s, q) via Avellaneda-Stoikov."""
        if self.cfg.max_position <= 0:
            return fair_value

        q_ratio = clamp(position / self.cfg.max_position, -1.0, 1.0)
        # Non-linear scaling: stronger mean-reverting pressure as position nears max limits
        q_eff = math.copysign(math.pow(abs(q_ratio), 1.2), q_ratio)

        vol_factor = 1.0 + (vol_bps / 20.0) * self.cfg.risk_aversion_gamma
        skew_bps = q_eff * self.cfg.skew_bps * vol_factor

        res_price = fair_value * (1.0 - skew_bps / 10000.0)
        return res_price

    def compute_quotes(
        self,
        state: MarketSnapshot,
        buy_blocked: bool,
        sell_blocked: bool,
        is_stressed: bool,
        dir_pressure_bps: float,
        buy_tox_bps: float,
        sell_tox_bps: float,
    ) -> Tuple[Optional[float], Optional[float], float, float]:
        """Calculates candidate bid and ask quotes respecting spread and inventory constraints."""
        mid = state.mid
        best_bid = state.best_bid
        best_ask = state.best_ask
        mkt_spread_bps = state.spread_bps
        tick = self.cfg.tick_size

        fair_val = self.compute_fair_value(state)
        res_px = self.compute_reservation_price(fair_val, state.bot_position, state.vol_bps)

        # Base half-spread adapted to current market spread (1-8 bps)
        half_spr_bps = clamp(mkt_spread_bps / 2.0, self.cfg.min_edge_bps, self.cfg.max_edge_bps)

        # Dynamic continuous directional fade
        bid_dist_bps = half_spr_bps + max(0.0, -dir_pressure_bps) + buy_tox_bps * 0.8
        ask_dist_bps = half_spr_bps + max(0.0, dir_pressure_bps) + sell_tox_bps * 0.8

        want_bid: Optional[float] = None
        want_ask: Optional[float] = None
        bid_qty = self.cfg.order_size
        ask_qty = self.cfg.order_size

        # --- BID QUOTE GENERATION --- #
        is_buy_unwind = (state.bot_position < 0.0)
        if is_buy_unwind:
            # Unwinding short: prioritize covering
            bid_qty = min(abs(state.bot_position), self.cfg.order_size)
            if is_stressed:
                cand_bid = best_bid
            else:
                cand_bid = min(best_ask - tick, res_px * (1.0 - self.cfg.min_edge_bps / 10000.0))
            cand_bid = min(cand_bid, best_ask - tick)
            want_bid = round(round(cand_bid / tick) * tick, 4)
        elif not buy_blocked and not is_stressed:
            # Normal liquidity provision with continuous fade
            model_bid = res_px * (1.0 - bid_dist_bps / 10000.0)
            cand_bid = min(best_bid, model_bid)
            cand_bid = min(cand_bid, best_ask - tick)
            want_bid = round(round(cand_bid / tick) * tick, 4)

        # --- ASK QUOTE GENERATION --- #
        is_sell_unwind = (state.bot_position > 0.0)
        if is_sell_unwind:
            # Unwinding long: prioritize exiting
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
            # Normal liquidity provision with continuous fade
            model_ask = res_px * (1.0 + ask_dist_bps / 10000.0)
            cand_ask = max(best_ask, model_ask)
            cand_ask = max(cand_ask, best_bid + tick)
            want_ask = round(round(cand_ask / tick) * tick, 4)

        # Cross Safety Guard: never cross quotes
        if want_bid is not None and want_ask is not None and want_bid >= want_ask:
            if state.bot_position > 0: want_bid = None
            elif state.bot_position < 0: want_ask = None
            else: want_bid = want_ask = None

        return want_bid, want_ask, bid_qty, ask_qty
