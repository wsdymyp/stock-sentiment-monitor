"""Unit tests for sentiment manipulation metrics, cooldowns, and end-to-end monitor."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.collectors import generate_mock_scenario
from src.config import MonitorConfig
from src.monitor import SentimentMonitor, get_adaptive_interval
from src.nlp_metrics import (
    compute_emotion_density,
    compute_homogeneity,
    compute_simhash,
    compute_uv_spike,
    evaluate_symbol_risk,
    hamming_distance,
)
from src.storage import SHANGHAI_TZ


class TestSentimentMonitor(unittest.TestCase):
    def test_simhash_near_duplicate(self) -> None:
        h1 = compute_simhash("算力总龙头今天主力锁仓，明天一字涨停，再不上车就踏空！")
        h2 = compute_simhash("算力总龙头今天主力锁仓，明天一字涨停，再不上车就踏空啦")
        h3 = compute_simhash("三季报营收和净利润同比增长25%，毛利率和现金流都很稳健")
        self.assertLessEqual(hamming_distance(h1, h2), 6)
        self.assertGreater(hamming_distance(h1, h3), 8)

    def test_anti_cold_start_uv_filter(self) -> None:
        cfg = MonitorConfig(min_uv_threshold=20)
        # Only 5 unique authors should not trigger UV spike even if baseline was 1.
        posts = [
            {
                "author_id": f"u_{i}",
                "content": "测试发帖",
                "platform": "guba",
                "published_at": datetime.now(SHANGHAI_TZ).isoformat(),
            }
            for i in range(5)
        ]
        uv, _, score = compute_uv_spike(posts, baseline_mean=1.0, baseline_std=1.0, config=cfg)
        self.assertEqual(uv, 5)
        self.assertEqual(score, 0.0)

    def test_homogeneity_and_burst_detection(self) -> None:
        cfg = MonitorConfig()
        bundle = generate_mock_scenario("high_bull_trap")
        homo_ratio, burst_ratio, samples, score = compute_homogeneity(bundle["posts"], cfg)
        self.assertGreater(homo_ratio, 0.45)
        self.assertGreater(burst_ratio, 0.5)
        self.assertGreater(score, 20.0)
        self.assertTrue(len(samples) >= 1)

    def test_official_announcement_whitelist_discount(self) -> None:
        cfg = MonitorConfig()
        bundle = generate_mock_scenario("high_bull_trap")
        _, _, _, score_with_ann = compute_emotion_density(bundle["posts"], True, cfg)
        _, _, _, score_no_ann = compute_emotion_density(bundle["posts"], False, cfg)
        self.assertGreater(score_no_ann, 0.0)
        self.assertLess(score_with_ann, score_no_ann)

    def test_regimes_and_cooldown_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cfg = MonitorConfig(db_path=db_path, risk_score_threshold=70.0, alert_cooldown_hours=6)
            monitor = SentimentMonitor(cfg)

            ref = datetime(2026, 11, 25, 20, 0, 0, tzinfo=SHANGHAI_TZ)

            # 1. High-position bull trap should score high and trigger alert.
            bull = generate_mock_scenario("high_bull_trap", reference_time=ref)
            res_bull = monitor.process_symbol(
                bull["market"].symbol,
                name=bull["market"].name,
                override_posts=bull["posts"],
                override_market=bull["market"],
                reference_time=ref,
            )
            self.assertEqual(res_bull.regime, "高位诱多出货型")
            self.assertGreaterEqual(res_bull.risk_score, 70.0)
            self.assertTrue(monitor.storage.is_in_cooldown(bull["market"].symbol, 6, reference_time=ref))

            # 2. Official news should stay below alert threshold & be classified as fundamental.
            news = generate_mock_scenario("official_news", reference_time=ref)
            res_news = monitor.process_symbol(
                news["market"].symbol,
                name=news["market"].name,
                override_posts=news["posts"],
                override_market=news["market"],
                reference_time=ref,
            )
            self.assertEqual(res_news.regime, "正常基本面催化")
            self.assertLess(res_news.risk_score, 70.0)

            # 3. Panic sell-off classification.
            panic = generate_mock_scenario("panic_selloff", reference_time=ref)
            res_panic = monitor.process_symbol(
                panic["market"].symbol,
                name=panic["market"].name,
                override_posts=panic["posts"],
                override_market=panic["market"],
                reference_time=ref,
            )
            self.assertEqual(res_panic.regime, "非理性恐慌踩踏型")

            # 4. Low-position seeding classification.
            low = generate_mock_scenario("low_seeding", reference_time=ref)
            res_low = monitor.process_symbol(
                low["market"].symbol,
                name=low["market"].name,
                override_posts=low["posts"],
                override_market=low["market"],
                reference_time=ref,
            )
            self.assertEqual(res_low.regime, "低位潜伏造势型")

    def test_adaptive_interval(self) -> None:
        cfg = MonitorConfig()
        pre_market = datetime(2026, 11, 25, 8, 30, tzinfo=SHANGHAI_TZ)
        intraday = datetime(2026, 11, 25, 11, 0, tzinfo=SHANGHAI_TZ)
        night = datetime(2026, 11, 25, 23, 30, tzinfo=SHANGHAI_TZ)
        self.assertEqual(get_adaptive_interval(cfg, pre_market), cfg.active_interval_sec)
        self.assertEqual(get_adaptive_interval(cfg, intraday), cfg.intraday_interval_sec)
        self.assertEqual(get_adaptive_interval(cfg, night), cfg.night_interval_sec)


if __name__ == "__main__":
    unittest.main()
