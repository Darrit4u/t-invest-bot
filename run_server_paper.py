"""CLI entrypoint for server paper-trading runtime."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.config_loader import ConfigError, ConfigLoader
from core.server_paper_runtime import run_server_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run server paper-trading runtime.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "config",
        help="Path to YAML configuration directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Path for log files",
    )
    parser.add_argument(
        "--run-seconds",
        type=int,
        default=0,
        help="Optional auto-stop timeout. 0 means run forever.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="How often to print runtime snapshots.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config files and exit without starting runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_config:
        try:
            ConfigLoader(args.config_dir).load()
        except ConfigError as exc:
            print(f"Config check failed: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print("Config check: OK")
        raise SystemExit(0)

    result = asyncio.run(
        run_server_paper(
            config_dir=args.config_dir,
            log_dir=args.log_dir,
            run_seconds=args.run_seconds,
            print_every=args.print_every,
        )
    )
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
