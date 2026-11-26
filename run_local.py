"""CLI entrypoint for GitHub Actions, local runs, or Colab verification."""

import argparse
import logging
import os

from src.config import MonitorConfig
from src.monitor import SentimentMonitor


def build_config_from_env(db_path: str) -> MonitorConfig:
    """Build MonitorConfig using environment variables (GitHub Secrets) when set."""
    raw_watchlist = os.getenv("WATCHLIST", "").strip()
    watchlist = [s.strip() for s in raw_watchlist.split(",") if s.strip()] if raw_watchlist else []

    raw_threshold = os.getenv("RISK_THRESHOLD", "").strip()
    risk_threshold = float(raw_threshold) if raw_threshold else 75.0

    raw_group_id = os.getenv("NONEBOT_GROUP_ID", "").strip()
    group_id = int(raw_group_id) if raw_group_id.isdigit() else 0

    raw_summary = os.getenv("SEND_TOP10_SUMMARY", "true").strip().lower()
    send_top10 = raw_summary not in ("0", "false", "no")

    return MonitorConfig(
        db_path=db_path,
        risk_score_threshold=risk_threshold,
        webhook_type=os.getenv("WEBHOOK_TYPE", "feishu").strip() or "feishu",
        webhook_url=os.getenv("WEBHOOK_URL", "").strip(),
        nonebot_api_url=os.getenv("NONEBOT_API_URL", "").strip(),
        nonebot_group_id=group_id,
        nonebot_token=os.getenv("NONEBOT_TOKEN", "").strip(),
        watchlist=watchlist,
        send_top10_summary=send_top10,
        focus_regime=os.getenv("FOCUS_REGIME", "低位潜伏造势型").strip(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Sentiment Manipulation Monitor")
    parser.add_argument("--db", default="data/monitor_data.db", help="SQLite DB path")
    parser.add_argument("--mock", action="store_true", help="Run once with mock scenarios")
    parser.add_argument("--loop", action="store_true", help="Run continuous 24/7 adaptive loop")
    parser.add_argument("--prune-days", type=int, default=14, help="Prune posts older than N days")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    db_dir = os.path.dirname(args.db)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    config = build_config_from_env(args.db)
    monitor = SentimentMonitor(config)

    if args.loop:
        monitor.run_forever(use_mock=args.mock)
    else:
        results = monitor.run_once(use_mock=args.mock)
        pruned = monitor.storage.prune_old_posts(keep_days=args.prune_days)
        if pruned:
            logging.info("Pruned %d expired posts (>%d days).", pruned, args.prune_days)
        for r in results:
            print(
                f"{r.name}({r.symbol}) -> Risk={r.risk_score:.1f} | "
                f"Regime={r.regime} | UV={r.uv_count} | Homo={r.homogeneity_ratio:.0%}"
            )


if __name__ == "__main__":
    main()
