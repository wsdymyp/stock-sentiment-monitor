"""NLP metrics, SimHash/cosine homogeneity, and RiskScore evaluation engine."""

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import (
    EXTREME_BEAR_WORDS,
    EXTREME_BULL_WORDS,
    FUNDAMENTAL_NEUTRAL_WORDS,
    MonitorConfig,
    PLATFORM_TIERS,
)

try:
    import jieba  # type: ignore
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Tokenize Chinese/English text using jieba if available, else char bigrams."""
    cleaned = _PUNCT_RE.sub("", text.lower())
    if not cleaned:
        return []
    if _HAS_JIEBA:
        tokens = [t.strip() for t in jieba.lcut(cleaned) if len(t.strip()) >= 1]
        if tokens:
            return tokens
    if len(cleaned) == 1:
        return [cleaned]
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def compute_simhash(text: str) -> str:
    """Compute a 64-bit SimHash hex string for fast near-duplicate detection."""
    tokens = tokenize(text)
    if not tokens:
        return "0" * 16
    weights = Counter(tokens)
    v = [0] * 64
    for token, weight in weights.items():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:8], byteorder="big")
        for i in range(64):
            v[i] += weight if (h >> i) & 1 else -weight
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= 1 << i
    return f"{fingerprint:016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Return Hamming distance between two 64-bit hex SimHashes."""
    return (int(hex1, 16) ^ int(hex2, 16)).bit_count()


def _cosine_similarity(c1: Counter, c2: Counter) -> float:
    """Compute cosine similarity between two sparse token frequency counters."""
    if not c1 or not c2:
        return 0.0
    common = set(c1.keys()) & set(c2.keys())
    if not common:
        return 0.0
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _char_ngrams(text: str, n: int = 2) -> Counter:
    """Extract character n-grams for robust AI-paraphrase similarity."""
    cleaned = _PUNCT_RE.sub("", text.lower())
    if len(cleaned) < n:
        return Counter([cleaned]) if cleaned else Counter()
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


@dataclass
class MarketContext:
    """Market price, turnover, and announcement context for a stock."""

    symbol: str
    name: str = ""
    pct_change_5d: float = 0.0
    turnover_rate: float = 0.0
    has_official_announcement: bool = False
    announcement_title: str = ""


@dataclass
class RiskAssessment:
    """Complete risk evaluation result for a monitored symbol."""

    symbol: str
    name: str
    risk_score: float
    regime: str
    uv_count: int
    post_count: int
    uv_zscore: float
    homogeneity_ratio: float
    burst_ratio: float
    emotion_density: float
    fundamental_ratio: float
    emotion_direction: str
    propagation_stage: str
    active_tiers: List[int]
    has_official_announcement: bool
    sample_phrases: List[str]

    def to_summary_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "risk_score": round(self.risk_score, 1),
            "regime": self.regime,
            "uv_count": self.uv_count,
            "post_count": self.post_count,
            "uv_zscore": round(self.uv_zscore, 2),
            "homogeneity_ratio": round(self.homogeneity_ratio, 3),
            "burst_ratio": round(self.burst_ratio, 3),
            "emotion_density": round(self.emotion_density, 3),
            "fundamental_ratio": round(self.fundamental_ratio, 3),
            "emotion_direction": self.emotion_direction,
            "propagation_stage": self.propagation_stage,
            "active_tiers": self.active_tiers,
            "has_official_announcement": self.has_official_announcement,
            "sample_phrases": self.sample_phrases,
        }


def compute_uv_spike(
    posts: List[Dict],
    baseline_mean: float,
    baseline_std: float,
    config: MonitorConfig,
) -> Tuple[int, float, float]:
    """Compute unique author count (UV), Z-Score, and sub-score (0-25)."""
    uv_count = len({p["author_id"] for p in posts})
    if uv_count < config.min_uv_threshold:
        zscore = 0.0 if baseline_mean <= 0 else (uv_count - baseline_mean) / max(baseline_std, 1.0)
        return uv_count, zscore, 0.0

    effective_std = max(baseline_std, math.sqrt(max(baseline_mean, 1.0)), 1.0)
    zscore = (uv_count - baseline_mean) / effective_std
    if zscore <= 1.0:
        score = 5.0
    else:
        score = min(25.0, 25.0 * (zscore / config.uv_zscore_trigger))
    return uv_count, zscore, score


def compute_homogeneity(
    posts: List[Dict],
    config: MonitorConfig,
) -> Tuple[float, float, List[str], float]:
    """
    Two-stage homogeneity (SimHash + bigram cosine) + 5-min burst detection.

    Returns (homogeneity_ratio, burst_ratio, sample_phrases, sub_score 0-30).
    """
    n = len(posts)
    if n < 5:
        return 0.0, 0.0, [], 0.0

    # Cap pairwise comparison window for O(1) predictable latency on Colab CPU.
    sample = posts[-120:] if n > 120 else posts
    m = len(sample)
    ngrams = [_char_ngrams(p["content"], 2) for p in sample]
    simhashes = [p.get("simhash") or compute_simhash(p["content"]) for p in sample]

    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(m):
        for j in range(i + 1, m):
            if hamming_distance(simhashes[i], simhashes[j]) <= config.simhash_max_hamming:
                union(i, j)
            elif _cosine_similarity(ngrams[i], ngrams[j]) >= config.cosine_similarity_threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(m):
        clusters[find(idx)].append(idx)

    # Posts belonging to any cluster of size >= 2 are considered homogeneous.
    clustered_indices = [idx for grp in clusters.values() if len(grp) >= 2 for idx in grp]
    homogeneity_ratio = len(clustered_indices) / m

    # Time-synchrony burst: max fraction of clustered posts within `burst_window_minutes`.
    burst_ratio = 0.0
    if len(clustered_indices) >= 3:
        times = sorted(
            datetime.fromisoformat(sample[idx]["published_at"]).timestamp()
            for idx in clustered_indices
        )
        window_sec = config.burst_window_minutes * 60
        left = 0
        max_in_win = 1
        for right in range(len(times)):
            while times[right] - times[left] > window_sec:
                left += 1
            max_in_win = max(max_in_win, right - left + 1)
        burst_ratio = max_in_win / len(clustered_indices)

    # Extract top cluster sample texts (up to 2).
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    sample_phrases = [
        sample[grp[0]]["content"][:80]
        for grp in sorted_clusters[:2]
        if len(grp) >= 2
    ]

    ratio_score = min(22.0, 22.0 * (homogeneity_ratio / max(config.homogeneity_ratio_trigger, 0.01)))
    burst_score = 8.0 * burst_ratio if homogeneity_ratio >= 0.15 else 0.0
    return homogeneity_ratio, burst_ratio, sample_phrases, min(30.0, ratio_score + burst_score)


def compute_emotion_density(
    posts: List[Dict],
    has_official_announcement: bool,
    config: MonitorConfig,
) -> Tuple[float, float, str, float]:
    """
    Compute extreme emotion density, fundamental ratio, direction, and sub-score (0-25).
    """
    if not posts:
        return 0.0, 1.0, "neutral", 0.0

    bull_hits = 0
    bear_hits = 0
    fundamental_hits = 0
    total_chars = 0

    for p in posts:
        text = p["content"]
        total_chars += max(len(text), 1)
        bull_hits += sum(text.count(w) for w in EXTREME_BULL_WORDS)
        bear_hits += sum(text.count(w) for w in EXTREME_BEAR_WORDS)
        fundamental_hits += sum(text.count(w) for w in FUNDAMENTAL_NEUTRAL_WORDS)

    total_emotion = bull_hits + bear_hits
    # Normalize per 100 Chinese characters.
    emotion_density = (total_emotion * 100.0) / max(total_chars, 100)
    fundamental_ratio = fundamental_hits / (total_emotion + fundamental_hits + 1)

    if bull_hits > bear_hits * 1.3:
        direction = "bull"
    elif bear_hits > bull_hits * 1.3:
        direction = "bear"
    else:
        direction = "mixed" if total_emotion > 0 else "neutral"

    density_factor = min(1.0, emotion_density / max(config.min_emotion_density * 10, 0.5))
    # Low fundamental support amplifies manipulation risk.
    lack_of_fundamentals = max(0.0, 1.0 - (fundamental_ratio / max(config.max_fundamental_ratio, 0.05)))
    raw_score = 25.0 * (0.65 * density_factor + 0.35 * lack_of_fundamentals)

    # Whitelist discount: official announcements explain high emotion.
    if has_official_announcement:
        raw_score *= 0.35

    return emotion_density, fundamental_ratio, direction, min(25.0, raw_score)


def compute_cross_platform_resonance(
    posts: List[Dict],
) -> Tuple[List[int], str, float]:
    """
    Track cross-platform propagation tiers and Lead-Lag stage (0-20 sub-score).
    """
    if not posts:
        return [], "isolated", 0.0

    tier_earliest: Dict[int, float] = {}
    for p in posts:
        tier = PLATFORM_TIERS.get(p["platform"], 2)
        ts = datetime.fromisoformat(p["published_at"]).timestamp()
        if tier not in tier_earliest or ts < tier_earliest[tier]:
            tier_earliest[tier] = ts

    active_tiers = sorted(tier_earliest.keys())
    if len(active_tiers) <= 1:
        stage = "early_seed" if active_tiers == [1] else "isolated"
        return active_tiers, stage, 4.0 if active_tiers == [1] else 0.0

    # Check Lead-Lag order: Tier 1 <= Tier 2 <= Tier 3
    if 3 in tier_earliest and (1 in tier_earliest or 2 in tier_earliest):
        stage = "mass_breakout"
        score = 20.0 if len(active_tiers) == 3 else 15.0
    elif 1 in tier_earliest and 2 in tier_earliest:
        stage = "retail_spread"
        score = 13.0 if tier_earliest[1] <= tier_earliest[2] else 10.0
    else:
        stage = "retail_spread"
        score = 10.0

    return active_tiers, stage, score


def classify_regime(
    market: MarketContext,
    emotion_direction: str,
    propagation_stage: str,
    config: MonitorConfig,
) -> str:
    """Classify manipulation/sentiment regime using price, announcements, and stage."""
    if market.has_official_announcement:
        return "正常基本面催化"

    if (
        market.pct_change_5d >= config.high_position_5d_pct
        and emotion_direction == "bull"
    ):
        return "高位诱多出货型"

    if (
        abs(market.pct_change_5d) <= config.low_position_5d_abs_pct
        and emotion_direction == "bull"
        and propagation_stage in ("early_seed", "retail_spread")
    ):
        return "低位潜伏造势型"

    if (
        market.pct_change_5d <= config.panic_drop_5d_pct
        and emotion_direction == "bear"
    ):
        return "非理性恐慌踩踏型"

    return "跨平台情绪共振异动"


def evaluate_symbol_risk(
    posts: List[Dict],
    market: MarketContext,
    baseline_mean: float,
    baseline_std: float,
    config: MonitorConfig,
) -> RiskAssessment:
    """Run all 4 optimized metrics and fuse into a 0-100 RiskAssessment."""
    uv_count, uv_zscore, uv_score = compute_uv_spike(
        posts, baseline_mean, baseline_std, config
    )
    homogeneity_ratio, burst_ratio, sample_phrases, homo_score = compute_homogeneity(
        posts, config
    )
    emotion_density, fundamental_ratio, direction, emo_score = compute_emotion_density(
        posts, market.has_official_announcement, config
    )
    active_tiers, stage, res_score = compute_cross_platform_resonance(posts)

    total_score = min(100.0, uv_score + homo_score + emo_score + res_score)
    regime = classify_regime(market, direction, stage, config)

    return RiskAssessment(
        symbol=market.symbol,
        name=market.name or market.symbol,
        risk_score=total_score,
        regime=regime,
        uv_count=uv_count,
        post_count=len(posts),
        uv_zscore=uv_zscore,
        homogeneity_ratio=homogeneity_ratio,
        burst_ratio=burst_ratio,
        emotion_density=emotion_density,
        fundamental_ratio=fundamental_ratio,
        emotion_direction=direction,
        propagation_stage=stage,
        active_tiers=active_tiers,
        has_official_announcement=market.has_official_announcement,
        sample_phrases=sample_phrases,
    )
