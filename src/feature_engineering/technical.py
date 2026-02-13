"""
テクニカル指標の計算
ta ライブラリを使って各種テクニカル指標を算出する。
"""
import numpy as np
import pandas as pd
import ta

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV データに対してテクニカル指標を計算し、特徴量として追加する。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV カラム (Open, High, Low, Close, Volume) を持つ DataFrame

    Returns
    -------
    pd.DataFrame
        テクニカル指標が追加された DataFrame
    """
    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    # ============================================================
    # リターン
    # ============================================================
    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)

    # 対数リターン
    result["log_return_1d"] = np.log(close / close.shift(1))

    # ============================================================
    # 移動平均 (SMA / EMA)
    # ============================================================
    for period in settings.SMA_PERIODS:
        result[f"sma_{period}"] = ta.trend.sma_indicator(close, window=period)
        # 終値との乖離率
        result[f"sma_{period}_deviation"] = (close - result[f"sma_{period}"]) / result[f"sma_{period}"]

    for period in settings.EMA_PERIODS:
        result[f"ema_{period}"] = ta.trend.ema_indicator(close, window=period)
        result[f"ema_{period}_deviation"] = (close - result[f"ema_{period}"]) / result[f"ema_{period}"]

    # ゴールデンクロス / デッドクロス シグナル
    if 5 in settings.SMA_PERIODS and 25 in settings.SMA_PERIODS:
        result["golden_cross_5_25"] = (
            (result["sma_5"] > result["sma_25"]) &
            (result["sma_5"].shift(1) <= result["sma_25"].shift(1))
        ).astype(int)
        result["dead_cross_5_25"] = (
            (result["sma_5"] < result["sma_25"]) &
            (result["sma_5"].shift(1) >= result["sma_25"].shift(1))
        ).astype(int)

    if 25 in settings.SMA_PERIODS and 75 in settings.SMA_PERIODS:
        result["golden_cross_25_75"] = (
            (result["sma_25"] > result["sma_75"]) &
            (result["sma_25"].shift(1) <= result["sma_75"].shift(1))
        ).astype(int)
        result["dead_cross_25_75"] = (
            (result["sma_25"] < result["sma_75"]) &
            (result["sma_25"].shift(1) >= result["sma_75"].shift(1))
        ).astype(int)

    # ============================================================
    # MACD
    # ============================================================
    macd = ta.trend.MACD(
        close,
        window_slow=settings.MACD_SLOW,
        window_fast=settings.MACD_FAST,
        window_sign=settings.MACD_SIGNAL,
    )
    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_histogram"] = macd.macd_diff()

    # MACDクロスシグナル
    result["macd_cross_up"] = (
        (result["macd"] > result["macd_signal"]) &
        (result["macd"].shift(1) <= result["macd_signal"].shift(1))
    ).astype(int)

    # ============================================================
    # RSI
    # ============================================================
    result["rsi"] = ta.momentum.rsi(close, window=settings.RSI_PERIOD)
    # RSIゾーン分類
    result["rsi_oversold"] = (result["rsi"] < 30).astype(int)
    result["rsi_overbought"] = (result["rsi"] > 70).astype(int)

    # ============================================================
    # ボリンジャーバンド
    # ============================================================
    bb = ta.volatility.BollingerBands(
        close,
        window=settings.BB_PERIOD,
        window_dev=settings.BB_STD,
    )
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_lower"] = bb.bollinger_lband()
    result["bb_middle"] = bb.bollinger_mavg()
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
    result["bb_position"] = (close - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])

    # ============================================================
    # ATR（Average True Range）
    # ============================================================
    result["atr"] = ta.volatility.average_true_range(
        high, low, close, window=settings.ATR_PERIOD
    )
    result["atr_ratio"] = result["atr"] / close  # 終値に対するATR比率

    # ============================================================
    # 出来高関連
    # ============================================================
    result["volume_sma_5"] = volume.rolling(5).mean()
    result["volume_sma_25"] = volume.rolling(25).mean()
    result["volume_ratio_5"] = volume / result["volume_sma_5"]
    result["volume_ratio_25"] = volume / result["volume_sma_25"]
    result["volume_change"] = volume.pct_change()

    # ============================================================
    # ストキャスティクス
    # ============================================================
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3
    )
    result["stoch_k"] = stoch.stoch()
    result["stoch_d"] = stoch.stoch_signal()

    # ============================================================
    # Williams %R
    # ============================================================
    result["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # ============================================================
    # ADX（Average Directional Index）
    # ============================================================
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    result["adx"] = adx.adx()
    result["adx_pos"] = adx.adx_pos()
    result["adx_neg"] = adx.adx_neg()

    # ============================================================
    # CCI（Commodity Channel Index）
    # ============================================================
    result["cci"] = ta.trend.cci(high, low, close, window=20)

    # ============================================================
    # ボラティリティ指標
    # ============================================================
    result["volatility_5d"] = result["log_return_1d"].rolling(5).std() * np.sqrt(252)
    result["volatility_20d"] = result["log_return_1d"].rolling(20).std() * np.sqrt(252)
    result["volatility_60d"] = result["log_return_1d"].rolling(60).std() * np.sqrt(252)

    # ============================================================
    # 価格パターン
    # ============================================================
    # 高値・安値からの位置
    result["high_20d"] = high.rolling(20).max()
    result["low_20d"] = low.rolling(20).min()
    result["price_position_20d"] = (close - result["low_20d"]) / (result["high_20d"] - result["low_20d"])

    result["high_60d"] = high.rolling(60).max()
    result["low_60d"] = low.rolling(60).min()
    result["price_position_60d"] = (close - result["low_60d"]) / (result["high_60d"] - result["low_60d"])

    # 52週高値/安値からの乖離
    result["high_252d"] = high.rolling(252).max()
    result["low_252d"] = low.rolling(252).min()
    result["from_52w_high"] = (close - result["high_252d"]) / result["high_252d"]
    result["from_52w_low"] = (close - result["low_252d"]) / result["low_252d"]

    # ============================================================
    # ローソク足パターン（簡易版）
    # ============================================================
    body = close - result["Open"]
    total_range = high - low
    result["candle_body_ratio"] = body / total_range.replace(0, np.nan)
    result["upper_shadow_ratio"] = (high - pd.concat([close, result["Open"]], axis=1).max(axis=1)) / total_range.replace(0, np.nan)
    result["lower_shadow_ratio"] = (pd.concat([close, result["Open"]], axis=1).min(axis=1) - low) / total_range.replace(0, np.nan)

    # 連続上昇/下落日数
    result["consecutive_up"] = _consecutive_count(result["return_1d"] > 0)
    result["consecutive_down"] = _consecutive_count(result["return_1d"] < 0)

    # ============================================================
    # 曜日・月の特徴
    # ============================================================
    if isinstance(result.index, pd.DatetimeIndex):
        result["day_of_week"] = result.index.dayofweek
        result["month"] = result.index.month
        result["is_month_end"] = result.index.is_month_end.astype(int)
        result["is_month_start"] = result.index.is_month_start.astype(int)

    logger.debug(f"テクニカル指標計算完了: {len(result.columns)} カラム")
    return result


def _consecutive_count(condition: pd.Series) -> pd.Series:
    """条件を満たす連続日数をカウントする"""
    groups = (~condition).cumsum()
    return condition.groupby(groups).cumsum()
