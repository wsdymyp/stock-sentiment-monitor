"""Alert formatting and delivery for Console, Feishu, DingTalk, WeCom, and NoneBot2."""

import logging
from typing import Dict, List

from .config import MonitorConfig
from .nlp_metrics import RiskAssessment

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def format_alert_message(assessment: RiskAssessment) -> str:
    """Format a concise decision card that activates System-2 rational thinking."""
    ann_status = "有官方公告支撑" if assessment.has_official_announcement else "无官方公告支撑 (需警惕小作文)"
    tiers_str = " -> ".join(f"T{t}" for t in assessment.active_tiers) or "单一平台"
    samples = "\n".join(f"  - {s}" for s in assessment.sample_phrases) or "  - 暂无高同质样本"

    return (
        f"【舆情操控风险预警】{assessment.name} ({assessment.symbol})\n"
        f"综合风险分: {assessment.risk_score:.1f}/100 | 场景定性: {assessment.regime}\n"
        f"独立作者(UV): {assessment.uv_count} (Z={assessment.uv_zscore:.2f}) | "
        f"同质化率: {assessment.homogeneity_ratio:.0%} (5分钟聚簇: {assessment.burst_ratio:.0%})\n"
        f"情绪密度: {assessment.emotion_density:.2f} ({assessment.emotion_direction}) | "
        f"基本面词占比: {assessment.fundamental_ratio:.0%}\n"
        f"传播路径: {tiers_str} ({assessment.propagation_stage}) | 公告校验: {ann_status}\n"
        f"高频同质话术摘要:\n{samples}\n"
        f"建议: 强制暂停10秒，核查F10公告与龙虎榜后再决策。"
    )


def format_top10_summary(
    assessments: List[RiskAssessment],
    top_n: int = 10,
    title: str = "低位潜伏高分",
) -> str:
    """Format a Top-N ranked sentiment risk summary (includes '预警' for webhook keyword filter)."""
    ranked = sorted(assessments, key=lambda r: r.risk_score, reverse=True)[:top_n]
    lines = [f"【{title} Top{len(ranked)} 播报｜预警参考】"]
    for idx, r in enumerate(ranked, start=1):
        ann = "有公告" if r.has_official_announcement else "无公告"
        lines.append(
            f"{idx}. {r.name}({r.symbol}) | 风险分:{r.risk_score:.1f} | "
            f"今日:{r.pct_change_today:+.1f}% | 5日:{r.pct_change_5d:+.1f}% | "
            f"UV:{r.uv_count} | 同质:{r.homogeneity_ratio:.0%} | {ann}"
        )
    return "\n".join(lines)


class AlertNotifier:
    """Dispatches formatted alerts and summaries to configured channels."""

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config

    def send(self, assessment: RiskAssessment) -> bool:
        return self.send_text(format_alert_message(assessment))

    def send_text(self, message: str) -> bool:
        """Send a plain text message to the configured channel."""
        logger.info("\n%s", message)

        if self.config.webhook_type == "nonebot" and self.config.nonebot_api_url:
            return self._send_nonebot(message)

        if self.config.webhook_url:
            return self._send_webhook(message)

        return True

    def _send_nonebot(self, message: str) -> bool:
        if not _HAS_REQUESTS or not self.config.nonebot_group_id:
            return False
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.nonebot_token:
            headers["Authorization"] = f"Bearer {self.config.nonebot_token}"
        payload = {
            "group_id": self.config.nonebot_group_id,
            "message": message,
        }
        try:
            resp = requests.post(
                self.config.nonebot_api_url,
                json=payload,
                headers=headers,
                timeout=8,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("NoneBot alert failed: %s", exc)
            return False

    def _send_webhook(self, message: str) -> bool:
        if not _HAS_REQUESTS:
            return False
        wtype = self.config.webhook_type.lower()
        if wtype == "feishu":
            payload = {"msg_type": "text", "content": {"text": message}}
        elif wtype in ("dingtalk", "wecom"):
            payload = {"msgtype": "text", "text": {"content": message}}
        else:
            payload = {"text": message}

        try:
            resp = requests.post(self.config.webhook_url, json=payload, timeout=8)
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Webhook (%s) alert failed: %s", wtype, exc)
            return False
