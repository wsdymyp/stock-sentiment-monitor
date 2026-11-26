# 财经流媒体心理暗示与情绪操控监控系统

> 把流媒体的“反复心理暗示”转化为可量化的异常信号，在“系统 1”冲动下单前，用客观数据强制唤醒“系统 2”理性核查。

支持 **GitHub Actions 24/7 无人值守全天运行（关机也能跑）**、**Google Colab** 与 **本地 CLI** 三种部署方式。

---

## 核心监控维度（RiskScore 0–100）

| 维度 | 对应心理暗示手法 | 量化与抗误报机制 |
| :--- | :--- | :--- |
| **1. 独立作者突增 (UV Z-Score)** | 曝光效应（高频洗脑） | 统计 2 小时独立作者数（UV）相对 7 天分时段 EWMA 基线的 $Z$ 分数，并设最小绝对阈值防冷门股误报。 |
| **2. 两级同质化与聚簇** | 水军刷屏 / AI 洗稿通稿 | `SimHash`（抓复制粘贴）+ `2-gram` 余弦相似度（抓换皮洗稿）+ 5 分钟发帖时间同步性检测。 |
| **3. 情绪密度与公告白名单** | 框架效应 / 情绪锚定 | 计算极端煽动词密度与基本面中性词占比；若存在交易所正式公告，自动大幅降权以防误报真实利好。 |
| **4. 跨平台传播梯度** | 资金合力造势 | 追踪 `T1(雪球/淘股吧) → T2(东财股吧) → T3(抖音/快手)` 传导阶段，区分“早期潜伏”与“高位破圈”。 |

结合 5 日涨跌幅与公告状态，系统自动将异常标的定性为：**高位诱多出货型**、**低位潜伏造势型**、**非理性恐慌踩踏型** 或 **正常基本面催化**。

---

## 方式一：GitHub Actions 24/7 无人值守部署（推荐）

通过 [monitor.yml](.github/workflows/monitor.yml) 每 15 分钟自动执行一次扫描，并将轻量化的 `data/monitor_data.db`（自动清理超过 14 天的旧帖子）提交回仓库保存 7 天基线与 6 小时告警冷却记录。

### 1. 仓库可见性选择
* **Public（公开仓库）**：GitHub Actions **完全免费且不限时长**，适合 24/7 全天候运行（所有 Webhook 地址均存放在加密的 Repository Secrets 中，不会泄露）。
* **Private（私有仓库）**：免费额度为每月 2,000 分钟；若全天每 15 分钟运行一次可能超出免费额度，建议将 `cron` 调整为交易时段（如 `"*/15 0-14 * * 1-5"`）。

### 2. 开启 Actions 写入权限（必须）
进入仓库 **Settings → Actions → General → Workflow permissions**，勾选：
* **Read and write permissions**（用于自动保存 `data/monitor_data.db` 状态）
* 点击 **Save**

### 3. 配置推送密钥（Repository Secrets）
进入仓库 **Settings → Secrets and variables → Actions → New repository secret**，按需添加：

| Secret 名称 | 必填 | 说明 |
| :--- | :---: | :--- |
| `WEBHOOK_TYPE` | 否 | `feishu`（默认）/ `dingtalk` / `wecom` / `nonebot` |
| `WEBHOOK_URL` | 推荐 | 飞书、钉钉或企业微信自定义机器人 Webhook 地址 |
| `WATCHLIST` | 否 | 逗号分隔的股票代码（如 `600519,300750`），留空则自动扫描东财人气前 10 |
| `RISK_THRESHOLD` | 否 | 预警阈值，默认 `75.0` |
| `NONEBOT_API_URL` | 否 | NoneBot2 HTTP 接口地址（如 `http://IP:5700/send_group_msg`） |
| `NONEBOT_GROUP_ID` | 否 | NoneBot2 接收群号 |
| `NONEBOT_TOKEN` | 否 | NoneBot2 Access Token |

> **飞书机器人安全设置提示**：在飞书群添加自定义机器人时，安全设置勾选 **「自定义关键词」** 并填入 `预警` 即可，无需开启签名校验。

### 4. 手动触发验证
1. 进入仓库顶部的 **Actions** 标签页 $\rightarrow$ 左侧选择 **24/7 Sentiment Manipulation Monitor**。
2. 点击右侧 **Run workflow** $\rightarrow$ 勾选 **`use_mock`** $\rightarrow$ 点击绿色的 **Run workflow**。
3. 等待约 30 秒，查看你的飞书/钉钉群或 Actions 日志，收到模拟预警卡片即表示全链路部署成功。之后无需再管，系统会按计划全天自动运行。

---

## 方式二：Google Colab 部署

上传整个项目至 Google Drive 的 `MyDrive/stock-sentiment-monitor`，打开 [colab_monitor.ipynb](colab_monitor.ipynb) 按 1–5 步运行即可（适合白天开机挂网页临时使用）。

## 方式三：本地运行与测试

```bash
pip install -r requirements.txt

# 运行单元测试
python3 -m unittest discover -s tests -v

# 离线模拟演练
python3 run_local.py --mock

# 单次真实扫描
python3 run_local.py
```

---

## 项目结构

```text
stock-sentiment-monitor/
├── .github/workflows/
│   ├── monitor.yml          # GitHub Actions 24/7 定时扫描与状态持久化
│   └── test.yml             # 自动单元测试 CI
├── data/                    # SQLite 状态文件目录 (monitor_data.db)
├── src/
│   ├── config.py            # 阈值、词典与平台层级配置
│   ├── storage.py           # SQLite 存储、EWMA 基线与冷却管理
│   ├── nlp_metrics.py       # UV Z-Score、SimHash/余弦同质化、情绪密度与定性
│   ├── collectors.py        # akshare 行情/公告校验 + 东财股吧采集 + 模拟数据
│   ├── notifier.py          # 飞书/钉钉/企业微信/NoneBot2 决策卡片推送
│   └── monitor.py           # 核心调度器与自适应轮询
├── tests/
│   └── test_monitor.py      # 单元测试
├── colab_monitor.ipynb      # Google Colab 运行笔记本
├── requirements.txt
└── run_local.py             # CLI 与 GitHub Actions 入口
```
