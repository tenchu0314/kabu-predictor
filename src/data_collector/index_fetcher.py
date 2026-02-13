"""
指数データ（日経225, ダウ, ドル円）の取得
"""
from typing import Optional

import pandas as pd
import yfinance as yf

from config import settings
from src.utils.helpers import save_dataframe, load_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_index_data(
    name: str,
    ticker: str,
    period: str = settings.HISTORY_PERIOD,
) -> Optional[pd.DataFrame]:
    """指数データを取得する"""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)

        if df.empty:
            logger.warning(f"{name} ({ticker}): データなし")
            return None

        # 不要カラム削除
        cols_to_drop = ["Dividends", "Stock Splits"]
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        # タイムゾーン除去
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"{name}: {len(df)} 日分のデータ取得")
        return df

    except Exception as e:
        logger.error(f"{name} ({ticker}): 取得エラー: {e}")
        return None


def fetch_all_indices() -> dict[str, pd.DataFrame]:
    """全指数データを取得・保存する"""
    logger.info("指数データ取得開始...")
    results = {}

    for name, ticker in settings.INDEX_TICKERS.items():
        df = fetch_index_data(name, ticker)
        if df is not None:
            filepath = settings.INDEX_DATA_DIR / f"{name}.csv"
            save_dataframe(df, filepath)
            results[name] = df

    logger.info(f"指数データ取得完了: {len(results)} 件")
    return results


def load_all_indices() -> dict[str, pd.DataFrame]:
    """保存済みの全指数データを読み込む"""
    results = {}
    for name in settings.INDEX_TICKERS.keys():
        filepath = settings.INDEX_DATA_DIR / f"{name}.csv"
        df = load_dataframe(filepath)
        if df is not None:
            results[name] = df
    return results
