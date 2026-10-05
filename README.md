# Arcus Level 9+ Institutional Market Maker: Hawkes & Stochastic Optimal Control Engine

A high-frequency quantitative market-making system engineered for tight-spread perpetual DEXes and limit order books.

---

## Architecture Overview

```
                               ┌─────────────────────────────────┐
                               │   Market Ingestion & Ticks/L2   │
                               └────────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌─────────────────────────────────┐                           ┌─────────────────────────────────┐
│   Continuous-Time Point Process │                           │   Cross-Exchange & Mark Alpha   │
│ - Multivariate Hawkes Engine    │                           │ - CEX Lead-Lag (Binance/Bybit)  │
│ - Trade & Cancel Self-Excitation│                           │ - Reference Basis Velocity      │
│ - Branching Ratio & Criticality │                           │ - Funding Carry & Micro-Drift   │
│ - Cascade & Sweep Detection     │                           └────────────────┬────────────────┘
└────────────────┬────────────────┘                                            │
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Cartea-Jaimungal Optimal Control│
                               │ - Analytical HJB Closed-Form    │
                               │ - Variance Risk Aversion (γ)    │
                               │ - Jump Adverse Selection (ΔS)   │
                               │ - Non-linear Inventory Bounds   │
                               └────────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌─────────────────────────────────┐                           ┌─────────────────────────────────┐
│    Active Delta Hedging Engine  │                           │   Selective-Touch Candidate Eval│
│ - Emergency Cascade Trigger     │                           │ - Touch vs 1-Tick vs Model      │
│ - Dynamic Tranche Sizing        │                           │ - Queue Hazard Fill Prob P(H)   │
│ - VWAP L2 Depth Walking         │                           │ - Empirical E[Markout|State]    │
│ - Inventory Variance Bound      │                           │ - One-Sided Steam Suppression   │
└─────────────────────────────────┘                           └─────────────────────────────────┘
```

---

## Key Advanced Strategies & Implementations

### 1. Multivariate Hawkes Point Process Engine (`hawkes.py`)
- Continuous-time mutually exciting point processes tracking aggressive buy trades, aggressive sell trades, and order cancellations/sweeps:
  $$\lambda_i(t) = \mu_i + \sum_{j=1}^M \sum_{t_{j,k} < t} lpha_{ij} e^{-eta (t - t_{j,k})}$$
- Recursive $O(1)$ state updates for sub-millisecond execution loops.
- Instantaneous Branching Ratio / Spectral Radius $ho(\Gamma)$:
  $$\Gamma_{ij} = rac{lpha_{ij}}{eta}$$
  Identifies when the market shifts from stable ($ho < 1$) to supercritical cascade regime ($ho \ge 1.0$).
- Dynamic Cascade Suppression: Detects order flow avalanches and immediately pulls quotes on the threatened side while widening quoting half-spreads.

### 2. Cartea-Jaimungal & Guéant-Tapia-Manziadi Stochastic Optimal Control (`optimal_control.py`)
- Rigorous mathematical optimization solving the Hamilton-Jacobi-Bellman (HJB) equations.
- Closed-form analytical reservation price with quadratic running inventory penalties and micro-drift alpha:
  $$R(s, q, t) = s - q \gamma \sigma^2 (T - t) + rac{lpha_t}{\kappa}$$
- Volatility-adaptive optimal half-spreads with adverse selection jump impact:
  $$\delta^{a*}(q) = \delta_0(\sigma) - q \cdot C(\sigma) + \Delta_{	ext{adverse}}^a$$
  $$\delta^{b*}(q) = \delta_0(\sigma) + q \cdot C(\sigma) + \Delta_{	ext{adverse}}^b$$
- Automatically widens spreads during volatility surges ($\sigma$) and skews asymmetrically to rapidly offload accumulated inventory without crossing mid-price.

### 3. Active Delta Hedging Engine (`hedger.py`)
- Active inventory variance protection preventing runaway drawdown during extended market trends.
- Dual-trigger thresholds:
  - Normal hedge threshold ($|q| > 75\%$ of max position).
  - Emergency cascade trigger ($|q| > 45\%$ of max position when Hawkes cascade is detected).
- Optimal Tranche Sizing: Automatically splits liquidation requirements into manageable chunks to eliminate catastrophic slippage.
- Execution Cost & Slippage Tracking: Ensures hedge benefits outweigh taker fees.

---

## Verification & Testing

Run all unit tests covering both the Level 7/8 foundation and the Level 9+ advanced engine:

```bash
python3 -m unittest test_level7.py test_advanced_v2.py
```

All 38 test suites pass with 100% success rate:
- `test_level7.py`: 31 tests covering orderbook intelligence, queue fill probability, VWAP walking, adaptive EV filtering, and positive spread capture.
- `test_advanced_v2.py`: 7 tests verifying Hawkes point process intensity/decay, spectral branching ratio, cascade detection, Cartea-Jaimungal closed-form optimal quotes, and Active Delta Hedger tranching.
