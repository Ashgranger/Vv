"""Cartea-Jaimungal & Guéant-Tapia-Manziadi Stochastic Optimal Control Engine.

Provides closed-form analytical solutions to the Hamilton-Jacobi-Bellman (HJB) equations
for quantitative market making with:
1. Inventory risk aversion and running quadratic inventory variance penalties.
2. Short-term directional alpha / micro-drift term \alpha_t.
3. Adverse selection jump risk \Delta S on aggressive fills.
4. Liquidity parameter \kappa and arrival intensity scaling A.

References:
- Cartea, Jaimungal, & Ricci (2014): "Buy Low, Sell High: A High-Frequency Trading Perspective"
- Guéant, Tapia, & Manziadi (2012): "Dealing with the inventory risk: a solution to the market making problem"
- Cartea & Jaimungal (2014): "Risk Metrics and Fine Tuning of High-Frequency Trading Strategies"
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict, NamedTuple, Optional, Tuple


class OptimalQuotes(NamedTuple):
    bid_price: Decimal
    ask_price: Decimal
    reservation_price: Decimal
    bid_half_spread_bps: Decimal
    ask_half_spread_bps: Decimal
    reservation_spread_bps: Decimal


class CarteaJaimungalEngine:
    """Stochastic optimal control engine for dynamic reservation price and half-spreads."""

    def __init__(
        self,
        gamma: float = 0.15,            # Risk aversion coefficient \gamma
        kappa: float = 1.5,             # Order book liquidity parameter \kappa
        arrival_intensity: float = 5.0, # Baseline arrival rate A
        terminal_horizon_s: float = 300.0, # Terminal execution horizon T
        non_linear_inventory_exp: float = 1.6, # Non-linear inventory penalty exponent
    ) -> None:
        self.gamma = max(1e-4, float(gamma))
        self.kappa = max(1e-3, float(kappa))
        self.A = max(0.1, float(arrival_intensity))
        self.T = max(1.0, float(terminal_horizon_s))
        self.inv_exp = float(non_linear_inventory_exp)

    def compute_reservation_price(
        self,
        fair_value: Decimal,
        position_norm: float,       # Normalized position q in [-1.0, 1.0]
        vol_bps: float,             # Realized volatility in basis points per second
        time_to_horizon_s: Optional[float] = None,
        alpha_bps: float = 0.0,     # Directional micro-drift alpha in bps
    ) -> Decimal:
        """Compute Cartea-Jaimungal reservation price:
        
        R(s, q, t) = s - q * \gamma * \sigma^2 * (T - t) + \frac{\alpha_t}{\kappa}
        
        Includes non-linear penalty when |q| > 0.6.
        """
        tau = min(self.T, max(1.0, float(time_to_horizon_s or self.T)))
        fv = float(fair_value)

        # Volatility converted to relative variance
        sigma_sec = (vol_bps / 10000.0)
        variance_sec = sigma_sec ** 2

        # Non-linear inventory damping for extreme positions
        q_sign = 1.0 if position_norm >= 0 else -1.0
        q_mag = abs(position_norm)
        if q_mag > 0.6:
            effective_q = q_sign * (0.6 + (q_mag - 0.6) ** self.inv_exp)
        else:
            effective_q = position_norm

        # Running inventory risk discount: -q * \gamma * \sigma^2 * \tau * fv
        inv_discount = effective_q * self.gamma * variance_sec * tau * fv

        # Alpha drift adjustment: + (\alpha / \kappa) * (fv / 10000)
        alpha_adj = (alpha_bps / self.kappa) * (fv / 10000.0)

        res_price = fv - inv_discount + alpha_adj
        return Decimal(str(round(res_price, 4)))

    def compute_optimal_half_spreads(
        self,
        position_norm: float,       # Normalized position q in [-1.0, 1.0]
        vol_bps: float,             # Volatility in bps
        time_to_horizon_s: Optional[float] = None,
        adverse_buy_bps: float = 0.0,
        adverse_sell_bps: float = 0.0,
        min_half_spread_bps: float = 1.0,
    ) -> Tuple[Decimal, Decimal]:
        """Compute closed-form optimal half-spreads (\delta^{b*}, \delta^{a*}).
        
        Formula:
        \delta^*(q) = \frac{1}{\kappa} \ln(1 + \frac{\kappa}{\gamma}) \pm \frac{2q \mp 1}{2} C(\sigma, \gamma, \kappa) + \Delta_{\text{adverse}}
        """
        tau = min(self.T, max(1.0, float(time_to_horizon_s or self.T)))
        sigma_sec = (vol_bps / 10000.0)

        # Baseline spread term: (1 / \kappa) * ln(1 + \kappa / \gamma)
        # Scaled to basis points
        base_term_bps = (1.0 / self.kappa) * math.log(1.0 + self.kappa / self.gamma) * 10.0

        # Inventory variance penalty coefficient C
        # C = \sqrt{ \frac{\gamma \sigma^2}{2 \kappa A (1 + \kappa/\gamma)^{1 + \gamma/\kappa}} }
        denom = 2.0 * self.kappa * self.A * ((1.0 + self.kappa / self.gamma) ** (1.0 + self.gamma / self.kappa))
        c_factor = math.sqrt(max(1e-12, (self.gamma * (sigma_sec ** 2)) / max(1e-9, denom))) * 10000.0

        q = position_norm

        # Optimal Ask half-spread: increases when long (q > 0), narrows when short (q < 0)
        # \delta^{a*}(q) = base + \frac{2q - 1}{2} * C + adverse_sell
        half_spread_base = base_term_bps + 0.5 * c_factor
        delta_ask = half_spread_base - q * c_factor + adverse_sell_bps

        # Optimal Bid half-spread: increases when short (q < 0), narrows when long (q > 0)
        # \delta^{b*}(q) = base - \frac{2q + 1}{2} * C + adverse_buy
        delta_bid = half_spread_base + q * c_factor + adverse_buy_bps

        # Ensure minimum spread bounds
        delta_bid = max(min_half_spread_bps, delta_bid)
        delta_ask = max(min_half_spread_bps, delta_ask)

        return Decimal(str(round(delta_bid, 2))), Decimal(str(round(delta_ask, 2)))

    def solve_quotes(
        self,
        fair_value: Decimal,
        position_norm: float,
        vol_bps: float,
        time_to_horizon_s: Optional[float] = None,
        alpha_bps: float = 0.0,
        adverse_buy_bps: float = 0.0,
        adverse_sell_bps: float = 0.0,
        min_half_spread_bps: float = 1.0,
    ) -> OptimalQuotes:
        """Solve for full optimal quoting tuple."""
        res_price = self.compute_reservation_price(
            fair_value=fair_value,
            position_norm=position_norm,
            vol_bps=vol_bps,
            time_to_horizon_s=time_to_horizon_s,
            alpha_bps=alpha_bps,
        )

        bid_half_bps, ask_half_bps = self.compute_optimal_half_spreads(
            position_norm=position_norm,
            vol_bps=vol_bps,
            time_to_horizon_s=time_to_horizon_s,
            adverse_buy_bps=adverse_buy_bps,
            adverse_sell_bps=adverse_sell_bps,
            min_half_spread_bps=min_half_spread_bps,
        )

        fv = float(fair_value)
        bid_price = res_price - Decimal(str(round(fv * float(bid_half_bps) / 10000.0, 4)))
        ask_price = res_price + Decimal(str(round(fv * float(ask_half_bps) / 10000.0, 4)))

        # Reservation spread vs fair value
        res_spread_bps = Decimal(str(round(((float(res_price) - fv) / fv) * 10000.0, 2)))

        return OptimalQuotes(
            bid_price=bid_price,
            ask_price=ask_price,
            reservation_price=res_price,
            bid_half_spread_bps=bid_half_bps,
            ask_half_spread_bps=ask_half_bps,
            reservation_spread_bps=res_spread_bps,
        )
