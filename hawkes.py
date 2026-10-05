r"""Multivariate Hawkes Point Process Engine for High-Frequency Microstructure Modeling.

Models the arrival intensities of aggressive buy trades, aggressive sell trades,
and order cancellations/sweeps using continuous-time mutually exciting point processes.

Mathematical Formulation:
    \lambda_i(t) = \mu_i + \sum_{j=1}^M \sum_{t_{j,k} < t} \alpha_{ij} e^{-\beta (t - t_{j,k})}

Where:
    - \mu_i is the baseline (exogenous) arrival intensity of event type i.
    - \alpha_{ij} is the excitation impact of event type j on event type i.
    - \beta is the exponential decay speed (half-life t_{1/2} = ln(2) / \beta).
    - \Gamma_{ij} = \alpha_{ij} / \beta is the branching matrix.
    - Spectral radius \rho(\Gamma) defines criticality:
        \rho < 1.0: Stable, stationary regime.
        \rho >= 1.0: Supercritical cascade regime (toxic cascade / flash crash danger).

Implements O(1) recursive state updates for ultra-low latency real-time execution.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict, List, Optional, Tuple


class HawkesProcessEngine:
    """Real-time multivariate Hawkes process engine for order flow toxicity detection."""

    EVENT_BUY = "buy"
    EVENT_SELL = "sell"
    EVENT_CANCEL = "cancel"

    def __init__(
        self,
        decay_beta: float = 2.0,           # Decay rate (half-life ~ 0.35s)
        baseline_mu: Optional[Dict[str, float]] = None,
        alpha_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        cascade_threshold_multiplier: float = 5.0,
        critical_branching_ratio: float = 0.85,
    ) -> None:
        self.beta = max(0.01, float(decay_beta))
        self.cascade_threshold_mult = float(cascade_threshold_multiplier)
        self.critical_branching = float(critical_branching_ratio)

        # Baseline intensities (exogenous Poisson rate)
        self.mu: Dict[str, float] = baseline_mu or {
            self.EVENT_BUY: 1.0,
            self.EVENT_SELL: 1.0,
            self.EVENT_CANCEL: 0.5,
        }

        # Alpha excitation matrix: (target_i, source_j) -> excitation magnitude
        # Self-excitation (momentum) and cross-excitation (pressure propagation)
        self.alpha: Dict[Tuple[str, str], float] = alpha_matrix or {
            (self.EVENT_BUY, self.EVENT_BUY): 0.70,     # Buy momentum
            (self.EVENT_BUY, self.EVENT_SELL): 0.15,    # Minor pull
            (self.EVENT_BUY, self.EVENT_CANCEL): 0.30,  # Ask cancellation sparks buying
            (self.EVENT_SELL, self.EVENT_SELL): 0.70,   # Sell momentum
            (self.EVENT_SELL, self.EVENT_BUY): 0.15,    # Minor pull
            (self.EVENT_SELL, self.EVENT_CANCEL): 0.30, # Bid cancellation sparks selling
            (self.EVENT_CANCEL, self.EVENT_BUY): 0.40,  # Aggressive buy causes ask cancel
            (self.EVENT_CANCEL, self.EVENT_SELL): 0.40, # Aggressive sell causes bid cancel
            (self.EVENT_CANCEL, self.EVENT_CANCEL): 0.20,
        }

        # Recursive state: R[i, j] tracks accumulated decayed excitation
        self.R: Dict[Tuple[str, str], float] = {
            (i, j): 0.0 for i in self.mu for j in self.mu
        }
        self.last_update_time: float = 0.0
        self.event_counts: Dict[str, int] = {k: 0 for k in self.mu}

    def _decay_state(self, now: float) -> None:
        """Decay recursive accumulator R to current timestamp `now`."""
        if self.last_update_time <= 0.0:
            self.last_update_time = now
            return

        dt = now - self.last_update_time
        if dt > 0.0:
            factor = math.exp(-self.beta * dt)
            for key in self.R:
                self.R[key] *= factor
            self.last_update_time = now

    def record_event(self, event_type: str, timestamp: float, size: float = 1.0) -> None:
        """Record an arrival event and advance Hawkes state."""
        if event_type not in self.mu:
            return

        # Advance state to timestamp
        self._decay_state(timestamp)

        # Normalized logarithmic sizing to prevent a single medium trade from triggering cascade
        clamped_size = max(0.5, min(2.5, 1.0 + 0.5 * math.log(max(0.1, float(size)))))

        for target_i in self.mu:
            key = (target_i, event_type)
            alpha_ij = self.alpha.get(key, 0.0)
            self.R[key] += alpha_ij * clamped_size

        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

    def get_intensity(self, event_type: str, timestamp: float) -> float:
        """Calculate instantaneous intensity lambda_i(t) at given timestamp."""
        if event_type not in self.mu:
            return 0.0

        dt = max(0.0, timestamp - self.last_update_time)
        factor = math.exp(-self.beta * dt)

        # Sum baseline + decayed excitations
        excitation = sum(
            self.R[(event_type, source_j)] * factor
            for source_j in self.mu
        )
        return self.mu[event_type] + excitation

    def get_all_intensities(self, timestamp: float) -> Dict[str, float]:
        """Return instantaneous intensities for all event channels."""
        return {k: self.get_intensity(k, timestamp) for k in self.mu}

    def get_branching_ratio(self) -> float:
        """Compute spectral radius rho(Gamma) of the branching matrix Gamma_ij = alpha_ij / beta.
        
        For a 2x2 or 3x3 matrix, computes largest real eigenvalue magnitude.
        """
        # Form matrix Gamma
        nodes = [self.EVENT_BUY, self.EVENT_SELL, self.EVENT_CANCEL]
        n = len(nodes)
        gamma = [[self.alpha.get((nodes[i], nodes[j]), 0.0) / self.beta for j in range(n)] for i in range(n)]

        # Power iteration to find spectral radius
        v = [1.0 / math.sqrt(n)] * n
        for _ in range(15):
            # w = gamma * v
            w = [sum(gamma[i][j] * v[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(x * x for x in w))
            if norm < 1e-9:
                return 0.0
            v = [x / norm for x in w]

        # Rayleigh quotient
        gv = [sum(gamma[i][j] * v[j] for j in range(n)) for i in range(n)]
        spectral_radius = sum(v[i] * gv[i] for i in range(n))
        return max(0.0, float(spectral_radius))

    def get_directional_bias(self, timestamp: float) -> float:
        """Returns directional toxicity imbalance in [-1.0, 1.0].
        
        Positive: Aggressive buy toxicity dominates.
        Negative: Aggressive sell toxicity dominates.
        """
        lam_buy = self.get_intensity(self.EVENT_BUY, timestamp)
        lam_sell = self.get_intensity(self.EVENT_SELL, timestamp)
        total = lam_buy + lam_sell + 1e-6
        return (lam_buy - lam_sell) / total

    def is_cascade_active(self, side: str, timestamp: float) -> bool:
        """Determines if a toxic cascade is active against quotes on `side`.
        
        If side == 'BUY', checks if aggressive SELL intensity has spiked above threshold.
        If side == 'SELL', checks if aggressive BUY intensity has spiked above threshold.
        """
        side_lower = side.lower()
        if side_lower == "buy":
            # Buy quotes are threatened by aggressive sell cascades
            lam_toxic = self.get_intensity(self.EVENT_SELL, timestamp)
            base = self.mu[self.EVENT_SELL]
        elif side_lower == "sell":
            # Sell quotes are threatened by aggressive buy cascades
            lam_toxic = self.get_intensity(self.EVENT_BUY, timestamp)
            base = self.mu[self.EVENT_BUY]
        else:
            return False

        # Exceeds threshold or branching ratio signals critical state
        is_spiking = lam_toxic >= (base * self.cascade_threshold_mult)
        is_critical = (self.get_branching_ratio() >= self.critical_branching) and (lam_toxic > base * 2.0)
        return is_spiking or is_critical

    def get_quote_skew_bps(self, side: str, timestamp: float) -> Decimal:
        """Calculates adaptive quote widening / skewing in basis points based on Hawkes intensity.
        
        Returns positive bps adjustment to push quote further away from mid.
        """
        side_lower = side.lower()
        if side_lower == "buy":
            lam_toxic = self.get_intensity(self.EVENT_SELL, timestamp)
            base = self.mu[self.EVENT_SELL]
        else:
            lam_toxic = self.get_intensity(self.EVENT_BUY, timestamp)
            base = self.mu[self.EVENT_BUY]

        ratio = lam_toxic / max(0.1, base)
        if ratio <= 1.0:
            return Decimal("0.0")

        # Non-linear spread widening for intense cascades
        skew = min(25.0, 1.5 * (ratio - 1.0) ** 1.3)
        return Decimal(str(round(skew, 2)))
