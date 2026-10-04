#!/usr/bin/env python3
"""CLI Entrypoint for Volatile Market Making Bot."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from config import Config
from environment import VolatileMarketEnv
from bot import VolatileMarketMaker

log = logging.getLogger("VolatileMM.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volatile Market Making Bot (1-8 bps Spread)")
    parser.add_argument("--mode", choices=["sim", "live"], default=None, help="Execution mode ('sim' or 'live')")
    parser.add_argument("--env-file", default=".env", help="Path to .env configuration file")
    parser.add_argument("--market", default=None, help="Target trading market symbol (e.g. BTC-USD)")
    parser.add_argument("--order-size", type=float, default=None, help="Order size in base asset")
    parser.add_argument("--max-pos", type=float, default=None, help="Maximum position limit")
    parser.add_argument("--min-edge", type=float, default=None, help="Minimum half-spread edge in bps")
    parser.add_argument("--max-edge", type=float, default=None, help="Maximum half-spread edge in bps")
    parser.add_argument("--skew-bps", type=float, default=None, help="Inventory skew in bps")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Simulation random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load configuration
    env_file = args.env_file if os.path.isfile(args.env_file) else None
    cfg = Config.from_env(env_file)

    # CLI overrides
    if args.mode: cfg.mode = args.mode
    if args.market: cfg.market = args.market
    if args.order_size: cfg.order_size = args.order_size
    if args.max_pos: cfg.max_position = args.max_pos
    if args.min_edge: cfg.min_edge_bps = args.min_edge
    if args.max_edge: cfg.max_edge_bps = args.max_edge
    if args.skew_bps: cfg.skew_bps = args.skew_bps
    if args.duration: cfg.sim_duration_s = args.duration
    if args.seed is not None: cfg.sim_seed = args.seed

    log.info("=" * 75)
    log.info("VOLATILE MARKET MAKER (1-8 BPS ADAPTIVE REGIME)")
    log.info(f"Mode: {cfg.mode.upper()} | Pair: {cfg.market} | Size: {cfg.order_size} | MaxPos: {cfg.max_position}")
    log.info(f"Spread Target: {cfg.target_spread_min_bps}-{cfg.target_spread_max_bps} bps | Edge: {cfg.min_edge_bps}-{cfg.max_edge_bps} bps | Skew: {cfg.skew_bps} bps")
    log.info("=" * 75)

    if cfg.mode == "sim":
        env = VolatileMarketEnv(cfg)

        class SimAdapter:
            def __init__(self, e: VolatileMarketEnv): self.e = e
            def place_order(self, s, p, q): return self.e.place_order(s, p, q)
            def cancel_order(self, oid): self.e.cancel_order(oid)
            def cancel_all_orders(self): self.e.cancel_all_orders()
            def get_fills(self): return self.e.bot_fills
            def step(self, dt): return self.e.step(dt)
            def get_analytics(self): return self.e.get_analytics()

        adapter = SimAdapter(env)
        bot = VolatileMarketMaker(cfg, adapter)
        analytics = bot.run_sim(cfg.sim_duration_s)

        print("\n" + "=" * 75)
        print("SIMULATION PERFORMANCE REPORT")
        print("=" * 75)
        print(f"{'Total Mark-to-Market PnL':<30} : ${analytics['total_pnl']:+.2f}")
        print(f"{'Realized PnL':<30} : ${analytics['realized_pnl']:+.2f}")
        print(f"{'Unrealized PnL':<30} : ${analytics['unrealized_pnl']:+.2f}")
        print(f"{'Total Fills Count':<30} : {analytics['total_fills']}")
        print(f"{'Ending Inventory':<30} : {analytics['final_position']:+.5f} BTC")
        print(f"{'Volume Traded (USD)':<30} : ${analytics['volume_traded_usd']:.2f}")
        print(f"{'Avg Markout at +1s (bps)':<30} : {analytics['avg_markout_1s_bps']:+.2f} bps")
        print(f"{'Avg Markout at +5s (bps)':<30} : {analytics['avg_markout_5s_bps']:+.2f} bps")
        print(f"{'Adverse Fill Ratio (+1s)':<30} : {analytics['adverse_fill_ratio_1s']*100:.1f}%")
        print(f"{'Adverse Fill Ratio (+5s)':<30} : {analytics['adverse_fill_ratio_5s']*100:.1f}%")
        print("=" * 75)

    elif cfg.mode == "live":
        log.info("Initializing Live Exchange Adapter...")
        # Import exchange connector from local environment
        sys.path.insert(0, "/working_dir")
        try:
            from bot import MarketMaker as LiveMarketMaker
            from config import Config as LiveConfig
            live_cfg = LiveConfig.from_env()
            live_bot = LiveMarketMaker(live_cfg)
            asyncio.run(live_bot.run())
        except Exception as e:
            log.error(f"Failed to start live mode: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
