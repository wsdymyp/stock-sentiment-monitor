"""Main monitoring orchestrator with adaptive polling for 24/7 Colab execution."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from .collectors import CommunityCollector, MarketContextCollector, generate_mock_scenario
from .config import MonitorConfig
from .nlp_metrics import MarketContext, RiskAssessment, evaluate_symbol_risk
from .notifier import AlertNotifier, format_top10_summary
from .storage import MonitorStorage, now_shanghai

logger = logging.getLogger(__name__)


def get_adaptive_interval(config: MonitorConfig, current_time: Optional[datetime] = None) -> int:
    """Return polling interval in seconds based on pre/post-market vs intraday vs night."""
    dt = current_time or now_shanghai()
    hhmm = dt.hour * 100 + dt.minute
    if (800 <= hhmm <= 930) or (1500 <= hhmm <= 2200):
        return config.active_interval_sec
    if 930 < hhmm < 1500:
        return config.intraday_interval_sec
    return config.night_interval_sec


class SentimentMonitor:
    """Coordinates collection, storage, NLP evaluation, and alerting."""

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.storage = MonitorStorage(config.db_path)
        self.market_collector = MarketContextCollector()
        self.community_collector = CommunityCollector()
        self.notifier = AlertNotifier(config)

    def process_symbol(
        self,
        symbol: str,
        name: str = "",
        override_posts: Optional[List[Dict]] = None,
        override_market: Optional[MarketContext] = None,
        reference_time: Optional[datetime] = None,
    ) -> RiskAssessment:
        """Run one full evaluation cycle for a single stock symbol."""
        ref = reference_time or now_shanghai()
        posts = (
            override_posts
            if override_posts is not None
            else self.community_collector.fetch_guba_posts(symbol)
        )
        if posts:
            self.storage.save_posts(posts)

        recent_posts = self.storage.get_recent_posts(
            symbol, hours=self.config.uv_window_hours, reference_time=ref
        )
        hour_slot = ref.hour // 2
        mean, std, _ = self.storage.get_uv_baseline(symbol, hour_slot)

        market = (
            override_market
            if override_market is not None
            else self.market_collector.get_context(symbol, name=name)
        )
        assessment = evaluate_symbol_risk(
            recent_posts, market, mean, std, self.config
        )

        # Update rolling EWMA baseline after evaluation.
        self.storage.update_uv_baseline(symbol, hour_slot, assessment.uv_count)

        if (
            assessment.risk_score >= self.config.risk_score_threshold
            and not self.storage.is_in_cooldown(
                symbol, self.config.alert_cooldown_hours, reference_time=ref
            )
        ):
            if self.notifier.send(assessment):
                self.storage.record_alert(
                    symbol=symbol,
                    regime=assessment.regime,
                    risk_score=assessment.risk_score,
                    summary=f"UV={assessment.uv_count}, Homo={assessment.homogeneity_ratio:.2f}",
                    reference_time=ref,
                )

        return assessment

    def run_once(self, use_mock: bool = False) -> List[RiskAssessment]:
        """Execute one full scan across watchlist or market hot symbols, returning Top-N by score."""
        results: List[RiskAssessment] = []
        if use_mock:
            for sc in ("high_bull_trap", "official_news", "panic_selloff", "low_seeding"):
                bundle = generate_mock_scenario(sc)
                m = bundle["market"]
                res = self.process_symbol(
                    m.symbol,
                    name=m.name,
                    override_posts=bundle["posts"],
                    override_market=m,
                )
                results.append(res)
        else:
            targets: List[Dict[str, str]] = [
                {"symbol": s, "name": s} for s in self.config.watchlist
            ]
            if not targets:
                if self.config.pool_mode == "main_board":
                    targets = self.community_collector.fetch_main_board_low_candidates(
                        limit=self.config.hot_scan_limit,
                        today_max_pct=self.config.low_position_today_max_pct,
                        max_60d_pct=self.config.low_position_60d_max_pct,
                    )
                else:
                    targets = self.community_collector.fetch_hot_symbols(
                        limit=self.config.hot_scan_limit
                    )

            for t in targets:
                try:
                    res = self.process_symbol(t["symbol"], name=t.get("name", ""))
                    results.append(res)
                    time.sleep(1.0)  # Polite delay between symbols
                except Exception as exc:
                    logger.error("Error processing %s: %s", t["symbol"], exc)

        ranked = sorted(results, key=lambda r: r.risk_score, reverse=True)
        if self.config.focus_regime:
            filtered = [r for r in ranked if r.regime == self.config.focus_regime]
            # Top up to top_n_output with other low-position, non-announcement Main Board stocks
            if len(filtered) < self.config.top_n_output and self.config.focus_regime == "低位潜伏造势型":
                seen_symbols = {r.symbol for r in filtered}
                extras = [
                    r
                    for r in ranked
                    if r.symbol not in seen_symbols
                    and r.price_fetched
                    and not r.has_official_announcement
                    and -5.0 <= r.pct_change_today <= self.config.low_position_today_max_pct
                    and abs(r.pct_change_5d) <= (self.config.low_position_5d_abs_pct + 2.0)
                ]
                filtered.extend(extras)
            title = "低位潜伏高分" if self.config.focus_regime == "低位潜伏造势型" else self.config.focus_regime
        else:
            filtered = ranked
            title = "舆情监控评分"

        top_results = filtered[: self.config.top_n_output]
        if top_results and self.config.send_top10_summary:
            self.notifier.send_text(
                format_top10_summary(top_results, top_n=self.config.top_n_output, title=title)
            )
        elif self.config.focus_regime:
            logger.info("No symbols matched focus_regime=%s in this scan.", self.config.focus_regime)
        return top_results

    def run_forever(self, use_mock: bool = False) -> None:
        """Run adaptive 24/7 loop with exception recovery for Colab."""
        logger.info("Starting 24/7 SentimentMonitor (db=%s)...", self.config.db_path)
        while True:
            try:
                assessments = self.run_once(use_mock=use_mock)
                logger.info(
                    "[%s] Scan finished: %d symbols evaluated.",
                    now_shanghai().strftime("%Y-%m-%d %H:%M:%S"),
                    len(assessments),
                )
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user.")
                break
            except Exception as exc:
                logger.exception("Unexpected loop error: %s", exc)

            sleep_sec = get_adaptive_interval(self.config)
            logger.info("Sleeping %d seconds until next cycle...", sleep_sec)
            time.sleep(sleep_sec)
