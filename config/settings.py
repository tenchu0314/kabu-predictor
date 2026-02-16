"""
Kabu Predictor - 設定ファイル
すべての設定値を一元管理する
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env ファイルを自動読み込み（cron等の環境でも確実に環境変数を設定）
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)


def _env(key: str, default: str = "") -> str:
    """環境変数を取得し、前後の空白・改行を除去する（CRLF対策）"""
    return os.environ.get(key, default).strip()


# ============================================================
# パス設定
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STOCK_DATA_DIR = DATA_DIR / "stocks"
INDEX_DATA_DIR = DATA_DIR / "indices"
MASTER_DATA_DIR = DATA_DIR / "master"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DAILY_REPORT_DIR = OUTPUT_DIR / "daily_reports"
LOG_DIR = PROJECT_ROOT / "logs"

# ディレクトリ作成
for d in [
    STOCK_DATA_DIR,
    INDEX_DATA_DIR,
    MASTER_DATA_DIR,
    MODEL_DIR,
    DAILY_REPORT_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# API Keys
# ============================================================
GEMINI_API_KEY = _env("GEMINI_API_KEY")

# ============================================================
# データ取得設定
# ============================================================
# 過去データ取得期間
HISTORY_PERIOD = "5y"

# 時価総額フィルタ（円）: 1000億円
MARKET_CAP_THRESHOLD = 100_000_000_000

# 指数ティッカー
INDEX_TICKERS = {
    "nikkei225": "^N225",
    "dow": "^DJI",
    "usdjpy": "JPY=X",
}

# yfinance リクエスト間隔（秒）- レートリミット対策
FETCH_INTERVAL = 0.5

# JPX 上場銘柄一覧 URL
JPX_STOCK_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# ============================================================
# 特徴量エンジニアリング設定
# ============================================================
# 移動平均期間
SMA_PERIODS = [5, 25, 75, 200]
EMA_PERIODS = [5, 25]

# RSI 期間
RSI_PERIOD = 14

# MACD パラメータ
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ボリンジャーバンド
BB_PERIOD = 20
BB_STD = 2

# ATR 期間
ATR_PERIOD = 14

# ローリング相関期間
CORRELATION_PERIOD = 20

# ============================================================
# 予測設定
# ============================================================
# 予測ホライゾン（営業日）と重み
PREDICTION_HORIZONS = {
    1: 0.30,  # 翌営業日 - デイトレ重視
    5: 0.30,  # 5営業日後 - 数日保有
    20: 0.25,  # 20営業日後 - 中期トレンド
    60: 0.15,  # 60営業日後 - 長期文脈
}

# ============================================================
# モデル設定
# ============================================================
# ウォークフォワード検証
WALK_FORWARD_TRAIN_MONTHS = 12  # 学習期間（月）
WALK_FORWARD_TEST_MONTHS = 1  # 検証期間（月）

# LightGBM デフォルトパラメータ
LGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "seed": 42,
}

# Optuna ハイパーパラメータ最適化
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 秒

# ============================================================
# スコアリング設定
# ============================================================
# 最終スコアの構成比
SCORE_WEIGHTS = {
    "prediction": 0.50,  # 予測スコア
    "fundamental": 0.25,  # ファンダメンタルスコア
    "risk_adjusted": 0.25,  # リスク調整スコア
    "overheat_penalty": 0.30,  # 過熱ペナルティ（基本スコアに対する最大減点率）
}

# トップN銘柄数
TOP_N = 10

# ============================================================
# スケジュール設定
# ============================================================
# 日次予測の実行時刻（JST）
DAILY_EXECUTION_HOUR = 6
DAILY_EXECUTION_MINUTE = 0

# モデル再学習の実行時刻（JST）- 毎週日曜 0:00
TRAIN_EXECUTION_HOUR = 0
TRAIN_EXECUTION_MINUTE = 0
TRAIN_EXECUTION_DAY = 6  # 0=月曜, 6=日曜

# 銘柄リスト更新曜日（0=月曜, 6=日曜）- モデル再学習と同時
STOCK_LIST_UPDATE_DAY = 6

# ============================================================
# Gemini API 設定
# ============================================================
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_MAX_TOKENS = 4096

# ============================================================
# Discord 通知設定
# ============================================================
DISCORD_BOT_TOKEN = _env("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = _env("DISCORD_CHANNEL_ID")

# ============================================================
# メール通知設定（オプション、Discord を使わない場合）
# ============================================================
# EMAIL_TO のみ設定 → ローカルの sendmail コマンドで送信
# SMTP_SERVER 等も設定 → 外部SMTPサーバー経由で送信
SMTP_SERVER = _env("SMTP_SERVER")
SMTP_PORT = int(_env("SMTP_PORT", "587"))
SMTP_USER = _env("SMTP_USER")
SMTP_PASSWORD = _env("SMTP_PASSWORD")
EMAIL_TO = _env("EMAIL_TO")

# ============================================================
# ログ設定
# ============================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE = LOG_DIR / "kabu_predictor.log"
