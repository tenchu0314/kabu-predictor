"""
JPX上場銘柄リストの取得・時価総額フィルタリング
"""
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import settings
from src.utils.helpers import (
    code_to_ticker,
    rate_limit_sleep,
    save_dataframe,
    load_dataframe,
    save_json,
    sanitize_filename,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_jpx_stock_list() -> pd.DataFrame:
    """
    JPXの上場銘柄一覧をダウンロードして保存する。
    xlsファイルをダウンロードしてDataFrameに変換する。
    """
    logger.info("JPX上場銘柄一覧をダウンロード中...")
    filepath = settings.MASTER_DATA_DIR / "jpx_stock_list.xls"

    try:
        response = requests.get(settings.JPX_STOCK_LIST_URL, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        df = pd.read_excel(filepath)
        logger.info(f"JPX銘柄一覧: {len(df)} 件取得")

        # CSV形式でも保存
        csv_path = settings.MASTER_DATA_DIR / "stock_list.csv"
        save_dataframe(df, csv_path, index=False)

        return df

    except Exception as e:
        logger.error(f"JPX銘柄一覧のダウンロードに失敗: {e}")
        # 既存のCSVがあれば読み込む
        csv_path = settings.MASTER_DATA_DIR / "stock_list.csv"
        if csv_path.exists():
            logger.info("既存のCSVファイルから読み込みます")
            return pd.read_csv(csv_path, encoding="utf-8-sig")
        raise


def get_stock_codes_from_jpx(df: pd.DataFrame) -> list[dict]:
    """
    JPXのDataFrameから銘柄コードと銘柄名を抽出する。
    ETF、REIT等は除外し、一般の株式のみを対象にする。
    """
    # JPXのカラム名はダウンロード時期により変動する可能性があるため
    # 柔軟にカラムを検索する
    code_col = None
    name_col = None
    market_col = None

    for col in df.columns:
        col_str = str(col)
        if "コード" in col_str and code_col is None:
            code_col = col
        elif "銘柄名" in col_str and name_col is None:
            name_col = col
        elif "市場" in col_str and market_col is None:
            market_col = col

    if code_col is None:
        # カラム名で見つからない場合は位置で推定
        logger.warning("カラム名から証券コードを特定できません。位置で推定します。")
        code_col = df.columns[0] if len(df.columns) > 0 else None
        name_col = df.columns[1] if len(df.columns) > 1 else None

    if code_col is None:
        raise ValueError("証券コードのカラムが見つかりません")

    stocks = []
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        # 4桁の数字コードのみ対象（ETF等の5桁コードを除外）
        if re.match(r'^\d{4}$', code):
            name = str(row[name_col]).strip() if name_col else ""
            stocks.append({
                "code": code,
                "name": name,
                "ticker": code_to_ticker(code),
            })

    logger.info(f"株式銘柄（4桁コード）: {len(stocks)} 件抽出")
    return stocks


def filter_by_market_cap(
    stocks: list[dict],
    threshold: float = settings.MARKET_CAP_THRESHOLD,
    batch_size: int = 50,
) -> list[dict]:
    """
    時価総額でフィルタリングする。
    yfinanceから時価総額を取得して閾値以上のもののみ返す。
    """
    logger.info(
        f"時価総額フィルタリング中... (閾値: {threshold/1e8:.0f}億円, 対象: {len(stocks)} 件)"
    )

    filtered = []
    failed = []

    for i, stock in enumerate(stocks):
        ticker_str = stock["ticker"]
        try:
            ticker = yf.Ticker(ticker_str)
            info = ticker.info
            market_cap = info.get("marketCap", 0)

            if market_cap and market_cap >= threshold:
                stock["market_cap"] = market_cap
                stock["sector"] = info.get("sector", "")
                stock["industry"] = info.get("industry", "")
                filtered.append(stock)
                logger.debug(
                    f"  ✓ {stock['code']} {stock['name']}: "
                    f"時価総額 {market_cap/1e8:.0f}億円"
                )

        except Exception as e:
            failed.append(ticker_str)
            logger.debug(f"  ✗ {ticker_str}: {e}")

        # 進捗表示
        if (i + 1) % 100 == 0:
            logger.info(f"  進捗: {i+1}/{len(stocks)} ({len(filtered)} 件通過)")

        rate_limit_sleep(settings.FETCH_INTERVAL)

    logger.info(f"時価総額フィルタ結果: {len(filtered)} 件（失敗: {len(failed)} 件）")

    # 結果を保存
    target_df = pd.DataFrame(filtered)
    save_dataframe(target_df, settings.MASTER_DATA_DIR / "target_stocks.csv", index=False)

    if failed:
        save_json(failed, settings.MASTER_DATA_DIR / "failed_tickers.json")

    return filtered


def load_target_stocks() -> Optional[pd.DataFrame]:
    """保存済みの対象銘柄リストを読み込む"""
    filepath = settings.MASTER_DATA_DIR / "target_stocks.csv"
    if filepath.exists():
        return pd.read_csv(filepath, encoding="utf-8-sig")
    return None


def update_stock_list() -> list[dict]:
    """
    銘柄リストを更新する一連の処理を実行する。
    1. JPXから銘柄一覧をダウンロード
    2. 株式銘柄を抽出
    3. 時価総額でフィルタリング
    """
    jpx_df = download_jpx_stock_list()
    stocks = get_stock_codes_from_jpx(jpx_df)
    target_stocks = filter_by_market_cap(stocks)
    return target_stocks


def get_target_stock_tickers() -> list[str]:
    """対象銘柄のティッカーリストを返す（保存済みがあればそれを使う）"""
    df = load_target_stocks()
    if df is not None and len(df) > 0:
        return df["ticker"].tolist()

    logger.info("対象銘柄リストが見つかりません。新規作成します。")
    stocks = update_stock_list()
    return [s["ticker"] for s in stocks]


def get_stock_folder_name(ticker: str) -> str:
    """ティッカーからフォルダ名を生成する（例: 7203_トヨタ自動車）"""
    df = load_target_stocks()
    if df is not None:
        row = df[df["ticker"] == ticker]
        if len(row) > 0:
            code = row.iloc[0]["code"]
            name = row.iloc[0]["name"]
            return sanitize_filename(f"{code}_{name}")

    code = ticker.replace(".T", "")
    return code
