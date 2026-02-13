"""
yfinanceを使った株価データ取得
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import settings
from src.data_collector.stock_list import get_stock_folder_name, get_target_stock_tickers
from src.utils.helpers import (
    rate_limit_sleep,
    save_dataframe,
    load_dataframe,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_price_history(
    ticker: str,
    period: str = settings.HISTORY_PERIOD,
) -> Optional[pd.DataFrame]:
    """
    指定ティッカーの株価履歴を取得する。
    OHLCV（始値、高値、安値、終値、出来高）を返す。
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)

        if df.empty:
            logger.warning(f"{ticker}: データなし")
            return None

        # 不要なカラムを削除（Dividends, Stock Splits）
        cols_to_drop = ["Dividends", "Stock Splits"]
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        # タイムゾーンを除去（CSV保存互換のため）
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.debug(f"{ticker}: {len(df)} 日分のデータ取得")
        return df

    except Exception as e:
        logger.error(f"{ticker}: 株価取得エラー: {e}")
        return None


def save_price_history(ticker: str, df: pd.DataFrame) -> Path:
    """株価履歴を銘柄フォルダに保存する"""
    folder_name = get_stock_folder_name(ticker)
    filepath = settings.STOCK_DATA_DIR / folder_name / "price_history.csv"
    save_dataframe(df, filepath)
    return filepath


def load_price_history(ticker: str) -> Optional[pd.DataFrame]:
    """保存済みの株価履歴を読み込む"""
    folder_name = get_stock_folder_name(ticker)
    filepath = settings.STOCK_DATA_DIR / folder_name / "price_history.csv"
    return load_dataframe(filepath)


def fetch_and_save_price(ticker: str) -> Optional[pd.DataFrame]:
    """株価データを取得して保存する"""
    df = fetch_price_history(ticker)
    if df is not None:
        save_price_history(ticker, df)
    return df


def fetch_all_prices(tickers: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    """
    全対象銘柄の株価データを取得・保存する。
    """
    if tickers is None:
        tickers = get_target_stock_tickers()

    logger.info(f"全銘柄の株価データ取得開始: {len(tickers)} 件")
    results = {}
    success = 0
    fail = 0

    for i, ticker in enumerate(tickers):
        df = fetch_and_save_price(ticker)
        if df is not None:
            results[ticker] = df
            success += 1
        else:
            fail += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  進捗: {i+1}/{len(tickers)} (成功: {success}, 失敗: {fail})")

        rate_limit_sleep(settings.FETCH_INTERVAL)

    logger.info(f"株価データ取得完了: 成功 {success} 件, 失敗 {fail} 件")
    return results


def load_all_prices(tickers: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    """保存済みの全銘柄の株価データを読み込む"""
    if tickers is None:
        tickers = get_target_stock_tickers()

    results = {}
    for ticker in tickers:
        df = load_price_history(ticker)
        if df is not None:
            results[ticker] = df

    logger.info(f"株価データ読み込み: {len(results)} 件")
    return results
