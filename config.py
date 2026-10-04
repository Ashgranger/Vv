"""Configuration management and environment variable loader."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v.strip())


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v.strip())


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Config:
    # Operating Mode & Connection
    mode: str = "sim"
    env_name: str = "testnet"
    wallet_address: str = ""
    api_signing_key: str = ""
    account_index: int = 0
    market: str = "BTC-USD"
    dry_run: bool = True

    # Order Sizing & Inventory Limits
    order_size: float = 0.005
    order_usd: float = 40.0
    max_position: float = 0.025
    soft_position: float = 0.015
    session_max_loss_usd: float = 50.0

    # Spread & Edge Tuning (1 - 8 bps Regime)
    min_edge_bps: float = 0.8
    max_edge_bps: float = 7.0
    target_spread_min_bps: float = 1.0
    target_spread_max_bps: float = 8.0
    skew_bps: float = 2.5
    risk_aversion_gamma: float = 0.15

    # Microstructure Alpha (OFI & Micro-price)
    enable_micro_intel: bool = True
    alpha_micro: float = 0.5
    alpha_obi: float = 0.6
    alpha_tfi: float = 0.7
    adverse_fade_mult: float = 1.0

    # Adverse Selection Guards
    trend_window_s: float = 2.0
    trend_pull_bps: float = 3.5
    burst_threshold: int = 2
    burst_window_s: float = 4.0
    burst_cooldown_s: float = 4.0
    min_depth_fade: float = 0.15

    # Execution Discipline
    retreat_drift_bps: float = 0.5
    advance_drift_bps: float = 1.2
    min_requote_interval_s: float = 0.15
    tick_size: float = 0.1
    step_size: float = 0.0001

    # Inventory Unwind & Stress
    max_hold_s: float = 30.0
    stress_loss_bps: float = 3.5
    exit_profit_bps: float = 0.3

    # Simulation-Specific Settings
    sim_duration_s: float = 180.0
    sim_noise_rate: float = 6.0
    sim_informed_rate: float = 0.4
    sim_jump_intensity: float = 0.06
    sim_base_vol_bps: float = 15.0
    sim_latency_s: float = 0.010
    sim_seed: int = 500

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        if env_path and os.path.isfile(env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                # Basic key=val line parser if python-dotenv is not installed
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            v = v.split("#")[0].strip()
                            os.environ[k.strip()] = v

        return cls(
            mode=_env_str("MODE", "sim").lower(),
            env_name=_env_str("ARCUS_ENV", "testnet").lower(),
            wallet_address=_env_str("ARCUS_WALLET_ADDRESS", ""),
            api_signing_key=_env_str("ARCUS_API_SIGNING_KEY", ""),
            account_index=_env_int("ARCUS_ACCOUNT_INDEX", 0),
            market=_env_str("MARKET", "BTC-USD"),
            dry_run=_env_bool("DRY_RUN", True),
            order_size=_env_float("ORDER_SIZE", 0.005),
            order_usd=_env_float("ORDER_USD", 40.0),
            max_position=_env_float("MAX_POSITION", 0.025),
            soft_position=_env_float("SOFT_POSITION", 0.015),
            session_max_loss_usd=_env_float("SESSION_MAX_LOSS_USD", 50.0),
            min_edge_bps=_env_float("MIN_EDGE_BPS", 0.8),
            max_edge_bps=_env_float("MAX_EDGE_BPS", 7.0),
            target_spread_min_bps=_env_float("TARGET_SPREAD_MIN_BPS", 1.0),
            target_spread_max_bps=_env_float("TARGET_SPREAD_MAX_BPS", 8.0),
            skew_bps=_env_float("SKEW_BPS", 2.5),
            risk_aversion_gamma=_env_float("RISK_AVERSION_GAMMA", 0.15),
            enable_micro_intel=_env_bool("ENABLE_MICRO_INTEL", True),
            alpha_micro=_env_float("ALPHA_MICRO", 0.5),
            alpha_obi=_env_float("ALPHA_OBI", 0.6),
            alpha_tfi=_env_float("ALPHA_TFI", 0.7),
            adverse_fade_mult=_env_float("ADVERSE_FADE_MULT", 1.0),
            trend_window_s=_env_float("TREND_WINDOW_S", 2.0),
            trend_pull_bps=_env_float("TREND_PULL_BPS", 3.5),
            burst_threshold=_env_int("BURST_THRESHOLD", 2),
            burst_window_s=_env_float("BURST_WINDOW_S", 4.0),
            burst_cooldown_s=_env_float("BURST_COOLDOWN_S", 4.0),
            min_depth_fade=_env_float("MIN_DEPTH_FADE", 0.15),
            retreat_drift_bps=_env_float("RETREAT_DRIFT_BPS", 0.5),
            advance_drift_bps=_env_float("ADVANCE_DRIFT_BPS", 1.2),
            min_requote_interval_s=_env_float("MIN_REQUOTE_INTERVAL_S", 0.15),
            tick_size=_env_float("TICK_SIZE", 0.1),
            step_size=_env_float("STEP_SIZE", 0.0001),
            max_hold_s=_env_float("MAX_HOLD_S", 30.0),
            stress_loss_bps=_env_float("STRESS_LOSS_BPS", 3.5),
            exit_profit_bps=_env_float("EXIT_PROFIT_BPS", 0.3),
            sim_duration_s=_env_float("SIM_DURATION_S", 180.0),
            sim_noise_rate=_env_float("SIM_NOISE_RATE", 6.0),
            sim_informed_rate=_env_float("SIM_INFORMED_RATE", 0.4),
            sim_jump_intensity=_env_float("SIM_JUMP_INTENSITY", 0.06),
            sim_base_vol_bps=_env_float("SIM_BASE_VOL_BPS", 15.0),
            sim_latency_s=_env_float("SIM_LATENCY_S", 0.010),
            sim_seed=_env_int("SIM_SEED", 500),
        )
