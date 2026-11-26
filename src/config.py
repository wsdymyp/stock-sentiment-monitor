"""Configuration, thresholds, and sentiment lexicons for the monitor."""

from dataclasses import dataclass, field
from typing import Dict, List


# Extreme hype / panic words used to anchor emotions instead of fundamentals.
EXTREME_BULL_WORDS: List[str] = [
    "十倍", "百倍", "史诗级", "十年一遇", "宇宙龙头", "总龙头",
    "踏空", "抢筹", "涨停潮", "连板", "闭眼买", "梭哈", "满仓",
    "起飞", "爆发", "翻倍", "主力锁仓", "机构疯抢", "明天一字",
    "最后上车", "财富密码", "暴富", "大肉", "必涨", "妖股",
]

EXTREME_BEAR_WORDS: List[str] = [
    "核按钮", "暴雷", "清仓式减持", "退市", "归零", "天地板",
    "崩盘", "踩踏", "腰斩", "割肉", "跑路", "完了", "死定了",
    "黑天鹅", "财务造假", "立案调查", "跌停潮", "快逃", "血亏",
]

# Neutral fundamental words indicating rational analysis.
FUNDAMENTAL_NEUTRAL_WORDS: List[str] = [
    "市盈率", "PE", "PB", "市净率", "营收", "净利润", "扣非",
    "毛利率", "现金流", "资产负债率", "研发投入", "分红", "股息率",
    "同比", "环比", "财报", "年报", "季报", "业绩预告", "估值",
    "产能", "订单", "市占率", "现金分红", "自由现金流",
]

# Platform tier mapping for Lead-Lag resonance tracking.
# Tier 1: Niche / early-signal communities
# Tier 2: Retail main hub
# Tier 3: Mass short-video / mainstream breaking-out
PLATFORM_TIERS: Dict[str, int] = {
    "xueqiu": 1,
    "taoguba": 1,
    "guba": 2,
    "douyin": 3,
    "kuaishou": 3,
    "toutiao": 3,
}


@dataclass
class MonitorConfig:
    """Runtime configuration for the sentiment manipulation monitor."""

    db_path: str = "monitor_data.db"
    risk_score_threshold: float = 75.0
    alert_cooldown_hours: int = 6

    # UV Spike parameters (anti-cold-start)
    uv_window_hours: int = 2
    baseline_days: int = 7
    min_uv_threshold: int = 20
    uv_zscore_trigger: float = 3.0

    # Homogeneity parameters
    simhash_max_hamming: int = 3
    cosine_similarity_threshold: float = 0.85
    burst_window_minutes: int = 5
    homogeneity_ratio_trigger: float = 0.30

    # Emotion density parameters
    min_emotion_density: float = 0.08
    max_fundamental_ratio: float = 0.25

    # Market regime thresholds
    high_position_5d_pct: float = 15.0
    low_position_5d_abs_pct: float = 6.0
    panic_drop_5d_pct: float = -10.0

    # Polling intervals (seconds) by market session
    active_interval_sec: int = 300      # 08:00-09:30, 15:00-22:00 (pre/post-market peak)
    intraday_interval_sec: int = 600    # 09:30-15:00
    night_interval_sec: int = 1800      # 22:00-08:00

    # Notifier settings
    webhook_url: str = ""
    webhook_type: str = "feishu"        # feishu | dingtalk | wecom | nonebot
    nonebot_api_url: str = ""           # e.g. http://127.0.0.1:5700/send_group_msg
    nonebot_group_id: int = 0
    nonebot_token: str = ""

    # Watchlist (empty = scan market hot symbols)
    watchlist: List[str] = field(default_factory=list)
    hot_scan_limit: int = 30
    top_n_output: int = 10
    send_top10_summary: bool = True
    focus_regime: str = "低位潜伏造势型"
