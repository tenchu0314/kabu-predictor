"""
ファンダメンタル指標の計算
yfinanceから取得した財務データを特徴量に変換する。
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.data_collector.financial_fetcher import load_stock_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_fundamental_features(ticker: str) -> dict:
    """
    ファンダメンタル特徴量を算出する。
    yfinanceのinfo情報を正規化して辞書形式で返す。

    Parameters
    ----------
    ticker : str
        yfinanceティッカーシンボル

    Returns
    -------
    dict
        ファンダメンタル特徴量の辞書
    """
    info = load_stock_info(ticker)
    if info is None:
        logger.warning(f"{ticker}: 基本情報なし")
        return {}

    features = {}

    # ============================================================
    # バリュエーション指標
    # ============================================================
    features["per_trailing"] = _safe_float(info.get("trailingPE"))
    features["per_forward"] = _safe_float(info.get("forwardPE"))
    features["pbr"] = _safe_float(info.get("priceToBook"))
    features["dividend_yield"] = _safe_float(info.get("dividendYield"))

    # PERの逆数（益回り）
    if features["per_trailing"] and features["per_trailing"] > 0:
        features["earnings_yield"] = 1.0 / features["per_trailing"]
    else:
        features["earnings_yield"] = None

    # ============================================================
    # 収益性指標
    # ============================================================
    features["roe"] = _safe_float(info.get("returnOnEquity"))
    features["roa"] = _safe_float(info.get("returnOnAssets"))
    features["operating_margin"] = _safe_float(info.get("operatingMargins"))
    features["profit_margin"] = _safe_float(info.get("profitMargins"))

    # ============================================================
    # 成長性指標
    # ============================================================
    features["revenue_growth"] = _safe_float(info.get("revenueGrowth"))
    features["earnings_growth"] = _safe_float(info.get("earningsGrowth"))

    # ============================================================
    # 財務健全性指標
    # ============================================================
    features["debt_to_equity"] = _safe_float(info.get("debtToEquity"))
    features["current_ratio"] = _safe_float(info.get("currentRatio"))
    features["quick_ratio"] = _safe_float(info.get("quickRatio"))

    # 自己資本比率（D/E比率から推定）
    if features["debt_to_equity"] is not None and features["debt_to_equity"] >= 0:
        features["equity_ratio"] = 1.0 / (1.0 + features["debt_to_equity"] / 100.0)
    else:
        features["equity_ratio"] = None

    # ============================================================
    # キャッシュフロー関連
    # ============================================================
    total_revenue = _safe_float(info.get("totalRevenue"))
    free_cashflow = _safe_float(info.get("freeCashflow"))
    total_cash = _safe_float(info.get("totalCash"))
    total_debt = _safe_float(info.get("totalDebt"))

    # FCFマージン
    if total_revenue and free_cashflow:
        features["fcf_margin"] = free_cashflow / total_revenue
    else:
        features["fcf_margin"] = None

    # ネットキャッシュ比率
    market_cap = _safe_float(info.get("marketCap"))
    if market_cap and total_cash is not None and total_debt is not None:
        features["net_cash_ratio"] = (total_cash - total_debt) / market_cap
    else:
        features["net_cash_ratio"] = None

    # ============================================================
    # リスク指標
    # ============================================================
    features["beta"] = _safe_float(info.get("beta"))

    # 52週高値/安値からの距離（infoベース）
    fifty_two_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    fifty_two_low = _safe_float(info.get("fiftyTwoWeekLow"))
    fifty_day_avg = _safe_float(info.get("fiftyDayAverage"))
    two_hundred_day_avg = _safe_float(info.get("twoHundredDayAverage"))

    if fifty_day_avg and two_hundred_day_avg:
        features["sma_50_200_ratio"] = fifty_day_avg / two_hundred_day_avg
    else:
        features["sma_50_200_ratio"] = None

    # 出来高変化
    avg_volume = _safe_float(info.get("averageVolume"))
    avg_volume_10d = _safe_float(info.get("averageVolume10days"))
    if avg_volume and avg_volume_10d and avg_volume > 0:
        features["volume_trend"] = avg_volume_10d / avg_volume
    else:
        features["volume_trend"] = None

    return features


def calculate_fundamental_score(features: dict) -> float:
    """
    ファンダメンタル特徴量からスコア（0〜1）を算出する。
    各指標を正規化してスコアリングする。
    """
    scores = []
    weights = []

    # PER: 低いほど割安（ただし負は除外）
    per = features.get("per_trailing")
    if per is not None and per > 0:
        # PER 5〜50 の範囲で正規化（低いほど高スコア）
        per_score = max(0, min(1, (50 - per) / 45))
        scores.append(per_score)
        weights.append(1.5)

    # PBR: 低いほど割安
    pbr = features.get("pbr")
    if pbr is not None and pbr > 0:
        pbr_score = max(0, min(1, (5 - pbr) / 4.5))
        scores.append(pbr_score)
        weights.append(1.0)

    # ROE: 高いほど良い
    roe = features.get("roe")
    if roe is not None:
        roe_score = max(0, min(1, roe / 0.3))  # 30%で満点
        scores.append(roe_score)
        weights.append(1.5)

    # 営業利益率
    op_margin = features.get("operating_margin")
    if op_margin is not None:
        op_score = max(0, min(1, op_margin / 0.3))
        scores.append(op_score)
        weights.append(1.0)

    # 売上成長率
    rev_growth = features.get("revenue_growth")
    if rev_growth is not None:
        growth_score = max(0, min(1, (rev_growth + 0.1) / 0.5))  # -10%〜+40%
        scores.append(growth_score)
        weights.append(1.0)

    # 自己資本比率
    equity_ratio = features.get("equity_ratio")
    if equity_ratio is not None:
        eq_score = max(0, min(1, equity_ratio))
        scores.append(eq_score)
        weights.append(0.8)

    # 配当利回り
    div_yield = features.get("dividend_yield")
    if div_yield is not None and div_yield >= 0:
        div_score = max(0, min(1, div_yield / 0.05))  # 5%で満点
        scores.append(div_score)
        weights.append(0.5)

    # FCFマージン
    fcf_margin = features.get("fcf_margin")
    if fcf_margin is not None:
        fcf_score = max(0, min(1, (fcf_margin + 0.05) / 0.25))
        scores.append(fcf_score)
        weights.append(0.7)

    if not scores:
        return 0.5  # データ不足の場合は中立値

    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    return weighted_score


def _safe_float(value) -> Optional[float]:
    """値を安全にfloatに変換する"""
    if value is None:
        return None
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None
