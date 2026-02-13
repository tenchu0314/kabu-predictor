"""
汎用ユーティリティ関数
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytz

JST = pytz.timezone("Asia/Tokyo")


def get_jst_now() -> datetime:
    """現在のJST時刻を返す"""
    return datetime.now(JST)


def is_trading_day(date: Optional[datetime] = None) -> bool:
    """営業日（平日）かどうかを判定する（祝日は未対応）"""
    if date is None:
        date = get_jst_now()
    return date.weekday() < 5


def get_last_trading_date() -> datetime:
    """直近の営業日を返す"""
    now = get_jst_now()
    # 市場開始前なら前日
    if now.hour < 9:
        now -= timedelta(days=1)
    while not is_trading_day(now):
        now -= timedelta(days=1)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def save_dataframe(df: pd.DataFrame, filepath: Path, index: bool = True) -> None:
    """DataFrameをCSVに保存する"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8-sig")


def load_dataframe(filepath: Path, index_col: Optional[int] = 0,
                   parse_dates: bool = True) -> Optional[pd.DataFrame]:
    """CSVからDataFrameを読み込む"""
    if not filepath.exists():
        return None
    return pd.read_csv(
        filepath, index_col=index_col, parse_dates=parse_dates,
        encoding="utf-8-sig"
    )


def save_json(data: Any, filepath: Path) -> None:
    """JSONファイルに保存する"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(filepath: Path) -> Optional[Any]:
    """JSONファイルを読み込む"""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def rate_limit_sleep(interval: float = 0.5) -> None:
    """レートリミット対策のスリープ"""
    time.sleep(interval)


def sanitize_filename(name: str) -> str:
    """ファイル名に使えない文字を置換する"""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def ticker_to_code(ticker: str) -> str:
    """yfinanceティッカー（例: 7203.T）から証券コード（例: 7203）を抽出"""
    return ticker.replace(".T", "")


def code_to_ticker(code: str) -> str:
    """証券コード（例: 7203）をyfinanceティッカー（例: 7203.T）に変換"""
    code = str(code).strip()
    if not code.endswith(".T"):
        return f"{code}.T"
    return code
