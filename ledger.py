"""Fill-driven accounting + Level 6 Online Learning of Adverse Selection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

import json
import logging
import os
import tempfile
from typing import Dict, Any
from utils import BPS, BUY, SELL, ZERO, ONE, clamp, fmt

log = logging.getLogger("ledger")


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


class OnlineLearner:
    """Dynamically adapts market making parameters based on live fill performance,
    orderbook flow predictive accuracy, and adverse selection markouts."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enable_online_learning", False))
        self.state_path = getattr(cfg, "learning_state_path", "learning_state.json")

        # Base configuration defaults
        self.base = {
            "min_edge_bps": Decimal(str(cfg.min_edge_bps)),
            "max_edge_bps": Decimal(str(cfg.max_edge_bps)),
            "skew_bps": Decimal(str(cfg.skew_bps)),
            "level_spacing_bps": Decimal(str(cfg.level_spacing_bps)),
            "level_size_mult": Decimal(str(cfg.level_size_mult)),
            "vol_k": Decimal(str(cfg.vol_k)),
            "tox_mult": Decimal(str(cfg.tox_mult)),
            "min_ev_bps": Decimal(str(cfg.min_ev_bps)),
            "obi_alpha": Decimal(str(cfg.obi_alpha)),
            "tfi_beta": Decimal(str(cfg.tfi_beta)),
            "fill_prob_kappa": Decimal(str(cfg.fill_prob_kappa)),
            "gamma_risk_aversion": Decimal(str(cfg.gamma_risk_aversion)),
            "regime_toxic_spread_mult": Decimal(str(cfg.regime_toxic_spread_mult)),
        }

        # Hard safety bounds [min_val, max_val]
        self.bounds = {
            "min_edge_bps": (Decimal("0.5"), Decimal("25.0")),
            "max_edge_bps": (Decimal("5.0"), Decimal("60.0")),
            "skew_bps": (Decimal("0.5"), Decimal("25.0")),
            "level_spacing_bps": (Decimal("1.0"), Decimal("20.0")),
            "level_size_mult": (Decimal("0.15"), Decimal("0.95")),
            "vol_k": (Decimal("0.1"), Decimal("3.0")),
            "tox_mult": (Decimal("0.2"), Decimal("4.0")),
            "min_ev_bps": (Decimal("0.05"), Decimal("3.0")),
            "obi_alpha": (Decimal("0.1"), Decimal("3.5")),
            "tfi_beta": (Decimal("0.1"), Decimal("3.5")),
            "fill_prob_kappa": (Decimal("0.05"), Decimal("1.0")),
            "gamma_risk_aversion": (Decimal("0.01"), Decimal("1.0")),
            "regime_toxic_spread_mult": (Decimal("1.1"), Decimal("3.0")),
        }

        # Current live parameters initialized to base values
        self.params: Dict[str, Decimal] = dict(self.base)

        # Learning metrics & stats
        self.n_markouts = 0
        self.n_toxic = 0
        self.n_benign = 0
        self.n_fills = 0
        self.total_learned_updates = 0

        if self.enabled:
            self.load()

    # --- Property Accessors for Engine --- #
    @property
    def min_edge_bps(self) -> Decimal:
        return self.params["min_edge_bps"] if self.enabled else self.base["min_edge_bps"]

    @property
    def max_edge_bps(self) -> Decimal:
        return self.params["max_edge_bps"] if self.enabled else self.base["max_edge_bps"]

    @property
    def skew_bps(self) -> Decimal:
        return self.params["skew_bps"] if self.enabled else self.base["skew_bps"]

    @property
    def level_spacing_bps(self) -> Decimal:
        return self.params["level_spacing_bps"] if self.enabled else self.base["level_spacing_bps"]

    @property
    def level_size_mult(self) -> Decimal:
        return self.params["level_size_mult"] if self.enabled else self.base["level_size_mult"]

    @property
    def vol_k(self) -> Decimal:
        return self.params["vol_k"] if self.enabled else self.base["vol_k"]

    @property
    def tox_mult(self) -> Decimal:
        return self.params["tox_mult"] if self.enabled else self.base["tox_mult"]

    @property
    def min_ev_bps(self) -> Decimal:
        return self.params["min_ev_bps"] if self.enabled else self.base["min_ev_bps"]

    @property
    def obi_alpha(self) -> Decimal:
        return self.params["obi_alpha"] if self.enabled else self.base["obi_alpha"]

    @property
    def tfi_beta(self) -> Decimal:
        return self.params["tfi_beta"] if self.enabled else self.base["tfi_beta"]

    @property
    def fill_prob_kappa(self) -> Decimal:
        return self.params["fill_prob_kappa"] if self.enabled else self.base["fill_prob_kappa"]

    @property
    def gamma_risk_aversion(self) -> Decimal:
        return self.params["gamma_risk_aversion"] if self.enabled else self.base["gamma_risk_aversion"]

    @property
    def regime_toxic_spread_mult(self) -> Decimal:
        return self.params["regime_toxic_spread_mult"] if self.enabled else self.base["regime_toxic_spread_mult"]

    def _clamp_all(self) -> None:
        for k in self.params:
            lo, hi = self.bounds[k]
            self.params[k] = clamp(self.params[k], lo, hi)
        if self.params["max_edge_bps"] < self.params["min_edge_bps"] + Decimal("2.0"):
            self.params["max_edge_bps"] = min(self.bounds["max_edge_bps"][1], self.params["min_edge_bps"] + Decimal("2.0"))

    def on_markout(self, m_bps: Decimal, side: str, tox_bps: Decimal) -> None:
        if not self.enabled:
            return

        self.n_markouts += 1
        self.total_learned_updates += 1

        if m_bps < 0:
            self.n_toxic += 1
            severity = min(Decimal("3.0"), abs(m_bps) / Decimal("5.0"))

            self.params["min_edge_bps"] += Decimal("0.25") * severity
            self.params["max_edge_bps"] += Decimal("0.50") * severity
            self.params["level_spacing_bps"] += Decimal("0.20") * severity
            self.params["level_size_mult"] -= Decimal("0.02") * severity
            self.params["tox_mult"] += Decimal("0.05") * severity
            self.params["min_ev_bps"] += Decimal("0.03") * severity
            self.params["regime_toxic_spread_mult"] += Decimal("0.04") * severity
            self.params["vol_k"] += Decimal("0.02") * severity
        else:
            self.n_benign += 1
            decay = Decimal("0.03")

            self.params["min_edge_bps"] -= (self.params["min_edge_bps"] - self.base["min_edge_bps"]) * decay
            self.params["max_edge_bps"] -= (self.params["max_edge_bps"] - self.base["max_edge_bps"]) * decay
            self.params["level_spacing_bps"] -= (self.params["level_spacing_bps"] - self.base["level_spacing_bps"]) * decay
            self.params["level_size_mult"] += (self.base["level_size_mult"] - self.params["level_size_mult"]) * decay
            self.params["tox_mult"] -= (self.params["tox_mult"] - self.base["tox_mult"]) * decay
            self.params["min_ev_bps"] -= (self.params["min_ev_bps"] - self.base["min_ev_bps"]) * decay
            self.params["regime_toxic_spread_mult"] -= (self.params["regime_toxic_spread_mult"] - self.base["regime_toxic_spread_mult"]) * decay
            self.params["vol_k"] -= (self.params["vol_k"] - self.base["vol_k"]) * decay

        self._clamp_all()
        self.save()

    def on_fill(self, side: str, price: Decimal, mid: Decimal, pos_usd: Decimal, hold_s: float) -> None:
        if not self.enabled:
            return

        self.n_fills += 1
        self.total_learned_updates += 1

        if mid and mid > 0:
            dist_bps = abs(price - mid) / mid * BPS
            if dist_bps > Decimal("1.5"):
                self.params["fill_prob_kappa"] -= Decimal("0.005")
            elif dist_bps < Decimal("0.2"):
                self.params["fill_prob_kappa"] += Decimal("0.002")

        max_pos = Decimal(str(self.cfg.max_position_usd))
        pos_ratio = abs(pos_usd) / max_pos if max_pos > 0 else ZERO

        if hold_s > 45.0 or pos_ratio > Decimal("0.6"):
            self.params["skew_bps"] += Decimal("0.25")
            self.params["gamma_risk_aversion"] += Decimal("0.015")
        elif pos_ratio < Decimal("0.15"):
            self.params["skew_bps"] -= (self.params["skew_bps"] - self.base["skew_bps"]) * Decimal("0.05")
            self.params["gamma_risk_aversion"] -= (self.params["gamma_risk_aversion"] - self.base["gamma_risk_aversion"]) * Decimal("0.05")

        self._clamp_all()
        self.save()

    def on_flow_correlation(self, obi: Decimal, tfi: Decimal, ret_bps: Decimal) -> None:
        if not self.enabled:
            return

        if abs(obi) > Decimal("0.2") and abs(ret_bps) > Decimal("0.1"):
            if (obi > 0 and ret_bps > 0) or (obi < 0 and ret_bps < 0):
                self.params["obi_alpha"] += Decimal("0.02")
            else:
                self.params["obi_alpha"] -= Decimal("0.02")

        if abs(tfi) > Decimal("0.2") and abs(ret_bps) > Decimal("0.1"):
            if (tfi > 0 and ret_bps > 0) or (tfi < 0 and ret_bps < 0):
                self.params["tfi_beta"] += Decimal("0.02")
            else:
                self.params["tfi_beta"] -= Decimal("0.02")

        self._clamp_all()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_updates": self.total_learned_updates,
            "markouts": self.n_markouts,
            "toxic_fills": self.n_toxic,
            "benign_fills": self.n_benign,
            "fills": self.n_fills,
            "params": {k: f"{v:.4f}" for k, v in self.params.items()},
        }

    def save(self, path: Optional[str] = None) -> bool:
        target_path = path or self.state_path
        if not target_path or target_path == os.devnull:
            return False

        data = {
            "version": 1,
            "enabled": self.enabled,
            "total_updates": self.total_learned_updates,
            "n_markouts": self.n_markouts,
            "n_toxic": self.n_toxic,
            "n_benign": self.n_benign,
            "n_fills": self.n_fills,
            "params": {k: str(v) for k, v in self.params.items()},
        }

        try:
            dir_name = os.path.dirname(os.path.abspath(target_path)) or "."
            fd, tmp_file = tempfile.mkstemp(dir=dir_name, prefix="learning_tmp_")
            with os.fdopen(fd, "w") as fp:
                json.dump(data, fp, indent=2)
            os.replace(tmp_file, target_path)
            return True
        except Exception:
            return False

    def load(self, path: Optional[str] = None) -> bool:
        target_path = path or self.state_path
        if not os.path.exists(target_path):
            return False

        try:
            with open(target_path, "r") as fp:
                data = json.load(fp)

            if isinstance(data, dict) and "params" in data:
                for k, v in data["params"].items():
                    if k in self.params:
                        self.params[k] = Decimal(str(v))
                self.n_markouts = int(data.get("n_markouts", 0))
                self.n_toxic = int(data.get("n_toxic", 0))
                self.n_benign = int(data.get("n_benign", 0))
                self.n_fills = int(data.get("n_fills", 0))
                self.total_learned_updates = int(data.get("total_updates", 0))
                self._clamp_all()
                return True
        except Exception:
            pass
        return False


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
        self.learner = OnlineLearner(cfg)

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
        self.learner.on_fill(side, price, mid, self.position * mid, self.hold_s(now))
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
            self.learner.on_markout(m_bps, f.side, self.tox_bps)

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
