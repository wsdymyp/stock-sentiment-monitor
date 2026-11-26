"""Market context (akshare) and community sentiment collectors with safe fallbacks."""

import logging
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .nlp_metrics import MarketContext, compute_simhash
from .storage import SHANGHAI_TZ, now_shanghai

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import akshare as ak  # type: ignore
    _HAS_AKSHARE = True
except ImportError:
    _HAS_AKSHARE = False


class MarketContextCollector:
    """Fetches 5-day price change, turnover, and official announcements via akshare."""

    def __init__(self) -> None:
        self._cached_notices = None
        self._cached_notice_date: str = ""

    def _get_today_notices(self):
        today = now_shanghai().strftime("%Y%m%d")
        if self._cached_notice_date != today:
            self._cached_notice_date = today
            try:
                self._cached_notices = ak.stock_notice_report(symbol="全部", date=today)
            except Exception as exc:
                logger.debug("akshare notice fetch failed: %s", exc)
                self._cached_notices = None
        return self._cached_notices

    def get_context(self, symbol: str, name: str = "") -> MarketContext:
        ctx = MarketContext(symbol=symbol, name=name or symbol)
        if not _HAS_AKSHARE:
            return ctx

        code = re.sub(r"\D", "", symbol)
        if not code:
            return ctx

        # Primary: Sina Finance K-line (avoids Eastmoney push2 rate limits on GitHub Actions)
        if _HAS_REQUESTS:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            sina_url = (
                "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                f"CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen=6"
            )
            try:
                resp = requests.get(sina_url, headers=CommunityCollector.HEADERS, timeout=8)
                resp.raise_for_status()
                bars = resp.json()
                if isinstance(bars, list) and len(bars) >= 2:
                    close_now = float(bars[-1]["close"])
                    close_prev_5d = float(bars[0]["close"])
                    close_prev_1d = float(bars[-2]["close"])
                    if close_prev_5d > 0:
                        ctx.pct_change_5d = ((close_now - close_prev_5d) / close_prev_5d) * 100.0
                    if close_prev_1d > 0:
                        ctx.pct_change_today = ((close_now - close_prev_1d) / close_prev_1d) * 100.0
                    ctx.price_fetched = True
            except Exception as exc:
                logger.debug("Sina kline fetch failed for %s: %s", symbol, exc)

        # Fallback: akshare Eastmoney K-line
        if not ctx.price_fetched and _HAS_AKSHARE:
            try:
                end_dt = now_shanghai()
                start_dt = end_dt - timedelta(days=14)
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_dt.strftime("%Y%m%d"),
                    end_date=end_dt.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                if df is not None and len(df) >= 2:
                    recent = df.tail(6)
                    close_now = float(recent.iloc[-1]["收盘"])
                    close_prev_5d = float(recent.iloc[0]["收盘"])
                    close_prev_1d = float(recent.iloc[-2]["收盘"])
                    if close_prev_5d > 0:
                        ctx.pct_change_5d = ((close_now - close_prev_5d) / close_prev_5d) * 100.0
                    if "涨跌幅" in recent.columns:
                        ctx.pct_change_today = float(recent.iloc[-1]["涨跌幅"])
                    elif close_prev_1d > 0:
                        ctx.pct_change_today = ((close_now - close_prev_1d) / close_prev_1d) * 100.0
                    if "换手率" in recent.columns:
                        ctx.turnover_rate = float(recent.iloc[-1]["换手率"])
                    ctx.price_fetched = True
            except Exception as exc:
                logger.debug("akshare price fetch failed for %s: %s", symbol, exc)

        try:
            notices = self._get_today_notices()
            if notices is not None and not notices.empty and "代码" in notices.columns:
                matched = notices[notices["代码"].astype(str) == code]
                if not matched.empty:
                    ctx.has_official_announcement = True
                    title_col = "公告标题" if "公告标题" in matched.columns else matched.columns[1]
                    ctx.announcement_title = str(matched.iloc[0][title_col])
        except Exception as exc:
            logger.debug("akshare notice check failed for %s: %s", symbol, exc)

        return ctx


_MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def is_main_board_a_share(symbol: str, name: str = "") -> bool:
    """Return True if symbol is a non-ST Shanghai/Shenzhen Main Board A-share."""
    code = re.sub(r"\D", "", symbol)
    if len(code) != 6 or not code.startswith(_MAIN_BOARD_PREFIXES):
        return False
    upper_name = name.upper()
    if "ST" in upper_name or "退" in name:
        return False
    return True


class CommunityCollector:
    """Collects public posts from Eastmoney Guba and Xueqiu with polite rate limits."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    def fetch_guba_posts(self, symbol: str, limit: int = 40) -> List[Dict]:
        """Fetch latest post titles from Eastmoney Guba."""
        if not _HAS_REQUESTS:
            return []

        code = re.sub(r"\D", "", symbol)
        if not code:
            return []

        url = f"https://guba.eastmoney.com/list,{code},f.html"
        posts: List[Dict] = []
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            # Extract article_list JSON embedded in Guba page when present.
            match = re.search(r"var\s+article_list\s*=\s*(\{.*?\});", resp.text, re.S)
            if not match:
                return []
            import json

            data = json.loads(match.group(1))
            items = data.get("re", [])[:limit]
            now_year = now_shanghai().year
            for item in items:
                title = str(item.get("post_title", "")).strip()
                if not title:
                    continue
                author = str(item.get("user_id") or item.get("user_nickname") or "anon")
                raw_time = str(item.get("post_publish_time") or "")
                pub_iso = self._parse_guba_time(raw_time, now_year)
                posts.append(
                    {
                        "symbol": symbol,
                        "platform": "guba",
                        "author_id": f"guba_{author}",
                        "content": title,
                        "simhash": compute_simhash(title),
                        "published_at": pub_iso,
                    }
                )
        except Exception as exc:
            logger.warning("Guba fetch failed for %s: %s", symbol, exc)
        return posts

    def fetch_main_board_low_candidates(
        self,
        limit: int = 40,
        today_max_pct: float = 4.5,
        max_60d_pct: float = 30.0,
    ) -> List[Dict[str, str]]:
        """
        Fast two-stage funnel over Main Board A-shares:
        1) Query Eastmoney Main Board snapshot (1-2 requests sorted by turnover rate)
        2) Fallback to Sina spot (ak.stock_zh_a_spot) or Eastmoney hot rank
        3) Keep non-ST Main Board stocks at low position with active turnover
        """
        candidates: List[tuple] = []

        # Primary: Sina Finance Main Board list sorted by turnoverratio (bypasses Eastmoney push2 limits)
        if _HAS_REQUESTS:
            for page in (1, 2, 3):
                sina_list_url = (
                    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                    f"Market_Center.getHQNodeData?page={page}&num=100&sort=turnoverratio&asc=0&node=hs_a"
                )
                try:
                    resp = requests.get(sina_list_url, headers=self.HEADERS, timeout=10)
                    resp.raise_for_status()
                    items = resp.json()
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        code = str(item.get("code", "")).strip()
                        name = str(item.get("name", "")).strip()
                        if not is_main_board_a_share(code, name):
                            continue
                        try:
                            pct_today = float(item.get("changepercent", 999))
                            amount = float(item.get("amount", 0) or 0)
                            turnover = float(item.get("turnoverratio", 0) or 0)
                        except (TypeError, ValueError):
                            continue

                        if not (-5.0 <= pct_today <= today_max_pct):
                            continue
                        if amount < 5e7 or turnover < 1.0:
                            continue
                        candidates.append((turnover, code, name))
                except Exception as exc:
                    logger.warning("Sina list page %d failed: %s", page, exc)

        # Secondary: Eastmoney clist API
        if not candidates and _HAS_REQUESTS:
            headers = {
                **self.HEADERS,
                "Referer": "https://quote.eastmoney.com/",
            }
            for page in (1, 2):
                url = (
                    "https://push2.eastmoney.com/api/qt/clist/get"
                    f"?pn={page}&pz=150&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
                    "&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:1+t:2"
                    "&fields=f12,f14,f3,f6,f8,f10,f24"
                )
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    diff = (resp.json().get("data") or {}).get("diff") or []
                    for item in diff:
                        code = str(item.get("f12", "")).strip()
                        name = str(item.get("f14", "")).strip()
                        if not is_main_board_a_share(code, name):
                            continue
                        try:
                            pct_today = float(item.get("f3", 999))
                            amount = float(item.get("f6", 0) or 0)
                            turnover = float(item.get("f8", 0) or 0)
                            vol_ratio = float(item.get("f10", 1.0) or 1.0)
                            pct_60d = float(item.get("f24", 0) or 0)
                        except (TypeError, ValueError):
                            continue

                        if not (-5.0 <= pct_today <= today_max_pct):
                            continue
                        if pct_60d > max_60d_pct:
                            continue
                        if amount < 5e7 or turnover < 1.0:
                            continue
                        candidates.append((turnover * max(vol_ratio, 0.5), code, name))
                except Exception as exc:
                    logger.warning("Eastmoney clist page %d failed: %s", page, exc)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            unique_out: List[Dict[str, str]] = []
            for _, code, name in candidates:
                if code not in seen:
                    seen.add(code)
                    unique_out.append({"symbol": code, "name": name})
                if len(unique_out) >= limit:
                    break
            return unique_out

        # Fallback: Eastmoney hot rank filtered to Main Board
        hot = self.fetch_hot_symbols(limit=max(limit, 50))
        return [h for h in hot if is_main_board_a_share(h["symbol"], h.get("name", ""))][:limit]

    def fetch_hot_symbols(self, limit: int = 10) -> List[Dict[str, str]]:
        """Fetch top hot A-share symbols via akshare if available."""
        if not _HAS_AKSHARE:
            return []
        try:
            df = ak.stock_hot_rank_em()
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(limit).iterrows():
                code = str(row.get("代码", "")).strip()
                name = str(row.get("股票名称", "")).strip()
                if code:
                    results.append({"symbol": code, "name": name})
            return results
        except Exception as exc:
            logger.warning("Hot symbols fetch failed: %s", exc)
            return []

    @staticmethod
    def _parse_guba_time(raw_time: str, default_year: int) -> str:
        raw_time = raw_time.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d %H:%M"):
            try:
                dt = datetime.strptime(raw_time, fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=default_year)
                return dt.replace(tzinfo=SHANGHAI_TZ).isoformat()
            except ValueError:
                continue
        return now_shanghai().isoformat()


def generate_mock_scenario(
    scenario: str = "high_bull_trap",
    reference_time: Optional[datetime] = None,
) -> Dict:
    """
    Generate realistic synthetic posts + MarketContext for offline/Colab testing.

    Scenarios: 'high_bull_trap', 'low_seeding', 'panic_selloff', 'official_news'
    """
    ref = reference_time or now_shanghai()
    rng = random.Random(42)

    if scenario == "high_bull_trap":
        symbol, name = "300999", "示例算力A"
        market = MarketContext(
            symbol=symbol,
            name=name,
            pct_change_5d=22.4,
            turnover_rate=18.5,
            has_official_announcement=False,
        )
        templates = [
            "算力总龙头今天主力锁仓，明天一字涨停，再不上车就踏空十倍行情！",
            "机构疯抢这只算力总龙头，主力锁仓明天一字板，再不买就彻底踏空！",
            "史诗级算力总龙头！机构疯抢锁仓，明天一字起飞，最后上车机会！",
        ]
        platforms = ["xueqiu", "guba", "douyin"]
        posts = []
        for i in range(32):
            tpl = templates[i % len(templates)] if i < 20 else f"大家怎么看{name}明天走势，感觉热度好高"
            # Burst 18 homogeneous posts within 4 minutes.
            offset_sec = rng.randint(0, 240) if i < 18 else rng.randint(300, 3600)
            ts = (ref - timedelta(seconds=offset_sec)).isoformat()
            plat = platforms[i % len(platforms)]
            posts.append(
                {
                    "symbol": symbol,
                    "platform": plat,
                    "author_id": f"{plat}_user_{i}",
                    "content": tpl,
                    "simhash": compute_simhash(tpl),
                    "published_at": ts,
                }
            )
        return {"market": market, "posts": posts}

    if scenario == "official_news":
        symbol, name = "600519", "示例白马B"
        market = MarketContext(
            symbol=symbol,
            name=name,
            pct_change_5d=6.2,
            turnover_rate=2.1,
            has_official_announcement=True,
            announcement_title="关于2026年第三季度业绩预增的公告",
        )
        posts = []
        for i in range(25):
            text = f"三季报营收和净利润同比增长25%，毛利率和现金流都很稳健，市盈率估值合理。"
            ts = (ref - timedelta(minutes=rng.randint(5, 90))).isoformat()
            posts.append(
                {
                    "symbol": symbol,
                    "platform": "xueqiu" if i % 2 == 0 else "guba",
                    "author_id": f"user_{i}",
                    "content": text,
                    "simhash": compute_simhash(text),
                    "published_at": ts,
                }
            )
        return {"market": market, "posts": posts}

    if scenario == "panic_selloff":
        symbol, name = "002888", "示例医药C"
        market = MarketContext(
            symbol=symbol,
            name=name,
            pct_change_5d=-14.5,
            turnover_rate=9.2,
            has_official_announcement=False,
        )
        templates = [
            "要暴雷退市了！主力核按钮跑路，赶紧割肉清仓不然归零！",
            "内部消息要暴雷！明天继续核按钮跌停潮，快逃割肉！",
        ]
        posts = []
        for i in range(26):
            text = templates[i % 2] if i < 18 else f"{name}跌得好惨，到底有没有问题"
            ts = (ref - timedelta(seconds=rng.randint(0, 200 if i < 16 else 3000))).isoformat()
            plat = ["guba", "douyin"][i % 2]
            posts.append(
                {
                    "symbol": symbol,
                    "platform": plat,
                    "author_id": f"{plat}_bear_{i}",
                    "content": text,
                    "simhash": compute_simhash(text),
                    "published_at": ts,
                }
            )
        return {"market": market, "posts": posts}

    # Default: low_seeding
    symbol, name = "000777", "示例低位D"
    market = MarketContext(
        symbol=symbol,
        name=name,
        pct_change_today=1.2,
        pct_change_5d=1.8,
        turnover_rate=3.4,
        price_fetched=True,
        has_official_announcement=False,
    )
    posts = []
    for i in range(24):
        text = "下一个十倍妖股正在底部启动，主力悄悄抢筹，即将爆发！" if i < 15 else f"关注{name}低位放量"
        ts = (ref - timedelta(seconds=rng.randint(0, 240 if i < 14 else 2400))).isoformat()
        plat = "xueqiu" if i < 12 else "guba"
        posts.append(
            {
                "symbol": symbol,
                "platform": plat,
                "author_id": f"{plat}_seed_{i}",
                "content": text,
                "simhash": compute_simhash(text),
                "published_at": ts,
            }
        )
    return {"market": market, "posts": posts}
