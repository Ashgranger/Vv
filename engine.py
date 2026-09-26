"""Level 7 Quantitative Market-Making Engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

from config import Config
from market import Market, MarketData
from ledger import Ledger
from utils import BPS, BUY, SELL, ZERO, ONE, clamp, q_down, q_up, fmt


@dataclass
class QuoteTarget:
    pair_index: int
    side: str
    price: Decimal
    qty: Decimal
    expected_value_bps: Decimal
    fill_probability: float
    is_exit_quote: bool


class MarketMakingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_fair_value(self, md: MarketData, now: float) -> Decimal:
        base_mid = md.mid
        if base_mid is None:
            return ZERO

        if not self.cfg.enable_orderbook_intel:
            return md.micro if (self.cfg.use_micro and md.micro) else base_mid

        micro = md.micro if md.micro else base_mid
        half_spr = (md.ask - md.bid) / Decimal("2") if (md.bid and md.ask) else ZERO

        obi = md.obi
        obi_shift = half_spr * obi * self.cfg.obi_alpha

        tfi = md.trade_flow_imbalance(10.0, now)
        tfi_shift = half_spr * tfi * self.cfg.tfi_beta

        fair_val = micro + obi_shift + tfi_shift
        return fair_val

    def fill_probability(self, distance_bps: Decimal) -> float:
        d = float(max(ZERO, distance_bps))
        kappa = float(self.cfg.fill_prob_kappa)
        return math.exp(-kappa * d)

    def expected_adverse_move(self, side: str, md: MarketData, ledger: Ledger, now: float) -> Decimal:
        base_tox = ledger.side_tox_bps(side)
        
        ret_5s = md.ret_bps(self.cfg.trend_window_s, now)
        momentum_risk = ZERO
        if side == BUY and ret_5s < 0:
            momentum_risk = abs(ret_5s) * self.cfg.trend_widen
        elif side == SELL and ret_5s > 0:
            momentum_risk = ret_5s * self.cfg.trend_widen

        tfi = md.trade_flow_imbalance(10.0, now)
        flow_risk = ZERO
        if side == BUY and tfi < Decimal("-0.2"):
            flow_risk = abs(tfi) * Decimal("1.5")
        elif side == SELL and tfi > Decimal("0.2"):
            flow_risk = tfi * Decimal("1.5")

        total_adverse = base_tox + momentum_risk + flow_risk
        return total_adverse

    def compute_reservation_price(self, fair_value: Decimal, position_usd: Decimal,
                                  vol_bps: Decimal) -> Decimal:
        if self.cfg.max_position_usd <= 0:
            return fair_value
        q = clamp(position_usd / self.cfg.max_position_usd, Decimal("-1"), Decimal("1"))
        
        q_eff = Decimal(str(math.copysign(math.pow(abs(float(q)), 1.3), float(q))))
        inv_skew_bps = q_eff * self.cfg.skew_bps
        
        if vol_bps > 0:
            inv_skew_bps += q_eff * vol_bps * self.cfg.gamma_risk_aversion

        res_price = fair_value * (ONE - inv_skew_bps / BPS)
        return res_price

    def generate_ladder_quotes(
        self,
        m: Market,
        md: MarketData,
        ledger: Ledger,
        now: float,
        buy_blocked: bool,
        sell_blocked: bool
    ) -> List[QuoteTarget]:
        if not md.bid or not md.ask or md.bid >= md.ask or not md.mid:
            return []

        mid = md.mid
        tick = m.tick_for(mid)
        step = m.step
        fair_val = self.compute_fair_value(md, now)
        pos_usd = ledger.position * mid
        res_price = self.compute_reservation_price(fair_val, pos_usd, md.vol_bps)

        regime = md.detect_regime(now, ledger.tox_bps)
        
        base_edge_bps = self.cfg.min_edge_bps + self.cfg.vol_k * md.vol_bps
        if self.cfg.enable_online_learning:
            base_edge_bps += self.cfg.tox_mult * ledger.tox_bps
        base_edge_bps = clamp(base_edge_bps, self.cfg.min_edge_bps, self.cfg.max_edge_bps)

        quotes: List[QuoteTarget] = []
        is_stressed = (pos_usd != 0 and (
            ledger.hold_s(now) > self.cfg.max_hold_s or
            (ledger.unrealized(mid) / abs(pos_usd) * BPS < -self.cfg.stress_loss_bps)
        ))

        total_levels = 1 + max(0, self.cfg.extra_levels)

        remaining_buy_usd = max(ZERO, self.cfg.max_position_usd - pos_usd)
        remaining_sell_usd = max(ZERO, self.cfg.max_position_usd + pos_usd)

        for k in range(total_levels):
            k_spacing = Decimal(str(k)) * self.cfg.level_spacing_bps
            level_edge = base_edge_bps + k_spacing
            
            size_mult = Decimal(str(math.pow(float(self.cfg.level_size_mult), k)))
            level_usd = max(self.cfg.order_usd * size_mult, m.min_notional)

            # --- BUY SIDE --- #
            is_unwind_buy = (pos_usd < 0)
            if is_unwind_buy:
                # UNWIND SHORT: Always active at touch to exit position and capture spread
                if k == 0:
                    if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                        cand_px = md.bid + tick
                    else:
                        cand_px = md.bid
                    cand_px = min(cand_px, md.ask - tick)
                    cand_px = q_down(cand_px, tick)
                    qty = q_down(abs(ledger.position), step)
                    if cand_px > ZERO and qty >= m.min_size:
                        quotes.append(QuoteTarget(
                            pair_index=0, side=BUY, price=cand_px, qty=qty,
                            expected_value_bps=Decimal("1.0"), fill_probability=0.9,
                            is_exit_quote=True
                        ))
            else:
                # ADDING LONG: Quote as long as inventory has room and side is not blocked
                can_add = (not buy_blocked) and (remaining_buy_usd >= level_usd)
                if can_add:
                    if k == 0 and self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                        cand_px = md.bid + tick
                    else:
                        cand_px = res_price * (ONE - level_edge / BPS)
                    
                    cand_px = min(cand_px, md.ask - tick)
                    cand_px = q_down(cand_px, tick)
                    
                    if cand_px > ZERO:
                        qty = q_down(level_usd / cand_px, step)
                        if qty >= m.min_size:
                            capture_bps = (fair_val - cand_px) / fair_val * BPS
                            adv_bps = self.expected_adverse_move(BUY, md, ledger, now)
                            dist_bps = (md.ask - cand_px) / mid * BPS
                            p_fill = self.fill_probability(dist_bps)
                            fee_bps = self.cfg.maker_fee_bps
                            inv_cost_bps = max(ZERO, pos_usd / self.cfg.max_position_usd) * self.cfg.skew_bps
                            
                            ev_bps = Decimal(str(p_fill)) * (capture_bps - adv_bps) - fee_bps - inv_cost_bps
                            
                            if (not self.cfg.enable_adaptive_ev) or (ev_bps >= self.cfg.min_ev_bps):
                                quotes.append(QuoteTarget(
                                    pair_index=k, side=BUY, price=cand_px, qty=qty,
                                    expected_value_bps=ev_bps, fill_probability=p_fill,
                                    is_exit_quote=False
                                ))
                                remaining_buy_usd -= (qty * cand_px)

            # --- SELL SIDE --- #
            is_unwind_sell = (pos_usd > 0)
            if is_unwind_sell:
                # UNWIND LONG: Always active at touch to exit position and capture spread
                if k == 0:
                    if self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                        cand_px = md.ask - tick
                    else:
                        cand_px = md.ask
                    cand_px = max(cand_px, md.bid + tick)
                    cand_px = q_up(cand_px, tick)
                    qty = q_down(abs(ledger.position), step)
                    if cand_px > ZERO and qty >= m.min_size:
                        quotes.append(QuoteTarget(
                            pair_index=0, side=SELL, price=cand_px, qty=qty,
                            expected_value_bps=Decimal("1.0"), fill_probability=0.9,
                            is_exit_quote=True
                        ))
            else:
                # ADDING SHORT: Quote as long as inventory has room and side is not blocked
                can_add = (not sell_blocked) and (remaining_sell_usd >= level_usd)
                if can_add:
                    if k == 0 and self.cfg.penny and (md.ask - md.bid) > Decimal("2") * tick:
                        cand_px = md.ask - tick
                    else:
                        cand_px = res_price * (ONE + level_edge / BPS)
                    
                    cand_px = max(cand_px, md.bid + tick)
                    cand_px = q_up(cand_px, tick)
                    
                    if cand_px > ZERO:
                        qty = q_down(level_usd / cand_px, step)
                        if qty >= m.min_size:
                            capture_bps = (cand_px - fair_val) / fair_val * BPS
                            adv_bps = self.expected_adverse_move(SELL, md, ledger, now)
                            dist_bps = (cand_px - md.bid) / mid * BPS
                            p_fill = self.fill_probability(dist_bps)
                            fee_bps = self.cfg.maker_fee_bps
                            inv_cost_bps = max(ZERO, -pos_usd / self.cfg.max_position_usd) * self.cfg.skew_bps
                            
                            ev_bps = Decimal(str(p_fill)) * (capture_bps - adv_bps) - fee_bps - inv_cost_bps
                            
                            if (not self.cfg.enable_adaptive_ev) or (ev_bps >= self.cfg.min_ev_bps):
                                quotes.append(QuoteTarget(
                                    pair_index=k, side=SELL, price=cand_px, qty=qty,
                                    expected_value_bps=ev_bps, fill_probability=p_fill,
                                    is_exit_quote=False
                                ))
                                remaining_sell_usd -= (qty * cand_px)

        return quotes
