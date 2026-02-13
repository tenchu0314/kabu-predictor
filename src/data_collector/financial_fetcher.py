"""
yfinanceを使った財務データ取得
（貸借対照表、損益計算書、キャッシュフロー、基本情報）
"""
from typing import Optional

import pandas as pd
import yfinance as yf

from config import settings
from src.data_collector.stock_list import get_stock_folder_name, get_target_stock_tickers
from src.utils.helpers import (
    rate_limit_sleep,
    save_dataframe,
    save_json,
    load_json,
    is_cache_fresh,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _df_to_json_safe_dict(df: pd.DataFrame) -> dict:
    """DataFrameをJSON保存可能なdictに変換する（Timestampキー対策）"""
    # カラム（決算日等）がTimestamp型の場合、文字列に変換
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    # インデックス（勘定科目名等）も文字列に
    df.index = [str(i) for i in df.index]
    return df.to_dict()


def fetch_financial_data(ticker: str) -> dict:
    """
    指定ティッカーの財務データをまとめて取得する。
    """
    try:
        t = yf.Ticker(ticker)
        result = {}

        # 基本情報
        try:
            info = t.info
            # 必要な情報のみ抽出
            key_info = {
                "marketCap": info.get("marketCap"),
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": info.get("priceToBook"),
                "dividendYield": info.get("dividendYield"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                "debtToEquity": info.get("debtToEquity"),
                "operatingMargins": info.get("operatingMargins"),
                "profitMargins": info.get("profitMargins"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "currentRatio": info.get("currentRatio"),
                "quickRatio": info.get("quickRatio"),
                "totalRevenue": info.get("totalRevenue"),
                "totalDebt": info.get("totalDebt"),
                "totalCash": info.get("totalCash"),
                "freeCashflow": info.get("freeCashflow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "beta": info.get("beta"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
                "twoHundredDayAverage": info.get("twoHundredDayAverage"),
                "averageVolume": info.get("averageVolume"),
                "averageVolume10days": info.get("averageVolume10days"),
            }
            result["info"] = key_info
        except Exception as e:
            logger.debug(f"{ticker}: info取得エラー: {e}")
            result["info"] = {}

        # 損益計算書
        try:
            financials = t.financials
            if financials is not None and not financials.empty:
                result["financials"] = _df_to_json_safe_dict(financials)
        except Exception as e:
            logger.debug(f"{ticker}: financials取得エラー: {e}")

        # 貸借対照表
        try:
            balance_sheet = t.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                result["balance_sheet"] = _df_to_json_safe_dict(balance_sheet)
        except Exception as e:
            logger.debug(f"{ticker}: balance_sheet取得エラー: {e}")

        # キャッシュフロー
        try:
            cashflow = t.cashflow
            if cashflow is not None and not cashflow.empty:
                result["cashflow"] = _df_to_json_safe_dict(cashflow)
        except Exception as e:
            logger.debug(f"{ticker}: cashflow取得エラー: {e}")

        return result

    except Exception as e:
        logger.error(f"{ticker}: 財務データ取得エラー: {e}")
        return {}


def save_financial_data(ticker: str, data: dict) -> None:
    """財務データを銘柄フォルダに保存する"""
    folder_name = get_stock_folder_name(ticker)
    base_dir = settings.STOCK_DATA_DIR / folder_name

    if "info" in data:
        save_json(data["info"], base_dir / "info.json")

    if "financials" in data:
        save_json(data["financials"], base_dir / "financials.json")

    if "balance_sheet" in data:
        save_json(data["balance_sheet"], base_dir / "balance_sheet.json")

    if "cashflow" in data:
        save_json(data["cashflow"], base_dir / "cashflow.json")


def load_stock_info(ticker: str) -> Optional[dict]:
    """保存済みの基本情報を読み込む"""
    folder_name = get_stock_folder_name(ticker)
    filepath = settings.STOCK_DATA_DIR / folder_name / "info.json"
    return load_json(filepath)


def fetch_all_financials(tickers: Optional[list[str]] = None) -> None:
    """全対象銘柄の財務データを取得・保存する"""
    if tickers is None:
        tickers = get_target_stock_tickers()

    logger.info(f"全銘柄の財務データ取得開始: {len(tickers)} 件")
    success = 0
    fail = 0
    skipped = 0

    for i, ticker in enumerate(tickers):
        # キャッシュチェック：info.jsonが新鮮ならスキップ
        folder_name = get_stock_folder_name(ticker)
        info_path = settings.STOCK_DATA_DIR / folder_name / "info.json"
        if is_cache_fresh(info_path):
            skipped += 1
            if (i + 1) % 50 == 0:
                logger.info(
                    f"  進捗: {i+1}/{len(tickers)} "
                    f"(取得: {success}, スキップ: {skipped}, 失敗: {fail})"
                )
            continue

        data = fetch_financial_data(ticker)
        if data:
            save_financial_data(ticker, data)
            success += 1
        else:
            fail += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"  進捗: {i+1}/{len(tickers)} "
                f"(取得: {success}, スキップ: {skipped}, 失敗: {fail})"
            )

        rate_limit_sleep(settings.FETCH_INTERVAL)

    logger.info(
        f"財務データ取得完了: 取得 {success} 件, "
        f"スキップ {skipped} 件, 失敗 {fail} 件"
    )
