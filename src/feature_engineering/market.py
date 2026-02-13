"""
マーケット連動指標の計算
日経225、ダウ、ドル円との相関・乖離を特徴量として算出する。
"""
import numpy as np
import pandas as pd

from config import settings
from src.data_collector.index_fetcher import load_all_indices
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_market_features(
    stock_df: pd.DataFrame,
    indices: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    個別銘柄のデータフレームにマーケット連動特徴量を追加する。

    Parameters
    ----------
    stock_df : pd.DataFrame
        個別銘柄のOHLCVデータ
    indices : dict, optional
        指数データの辞書 {"nikkei225": df, "dow": df, "usdjpy": df}
        Noneの場合は保存済みデータを読み込む

    Returns
    -------
    pd.DataFrame
        マーケット連動特徴量が追加されたDataFrame
    """
    if indices is None:
        indices = load_all_indices()

    result = stock_df.copy()
    stock_returns = result["Close"].pct_change()

    # ============================================================
    # 日経225 との連動指標
    # ============================================================
    if "nikkei225" in indices:
        nikkei = indices["nikkei225"]
        nikkei_close = _align_index(nikkei["Close"], result.index)
        nikkei_returns = nikkei_close.pct_change()

        # 日経リターン
        result["nikkei_return_1d"] = nikkei_returns
        result["nikkei_return_5d"] = nikkei_close.pct_change(5)
        result["nikkei_return_20d"] = nikkei_close.pct_change(20)

        # ローリング相関
        result["corr_nikkei_20d"] = stock_returns.rolling(
            settings.CORRELATION_PERIOD
        ).corr(nikkei_returns)
        result["corr_nikkei_60d"] = stock_returns.rolling(60).corr(nikkei_returns)

        # 相対強度（個別銘柄 vs 日経225）
        result["relative_strength_nikkei_5d"] = (
            result["Close"].pct_change(5) - nikkei_close.pct_change(5)
        )
        result["relative_strength_nikkei_20d"] = (
            result["Close"].pct_change(20) - nikkei_close.pct_change(20)
        )

        # 日経225のテクニカル状態
        nikkei_sma25 = nikkei_close.rolling(25).mean()
        result["nikkei_above_sma25"] = (nikkei_close > nikkei_sma25).astype(int)

    # ============================================================
    # ダウ との連動指標
    # ============================================================
    if "dow" in indices:
        dow = indices["dow"]
        dow_close = _align_index(dow["Close"], result.index)
        dow_returns = dow_close.pct_change()

        result["dow_return_1d"] = dow_returns
        result["dow_return_5d"] = dow_close.pct_change(5)

        # ローリング相関
        result["corr_dow_20d"] = stock_returns.rolling(
            settings.CORRELATION_PERIOD
        ).corr(dow_returns)

        # 前日のダウリターン（日本市場への影響を測定）
        # ダウは日本時間の早朝に終了するため、前日のダウが当日の日本株に影響
        result["dow_prev_return"] = dow_returns.shift(1)

    # ============================================================
    # ドル円 との連動指標
    # ============================================================
    if "usdjpy" in indices:
        usdjpy = indices["usdjpy"]
        usdjpy_close = _align_index(usdjpy["Close"], result.index)
        usdjpy_returns = usdjpy_close.pct_change()

        result["usdjpy_rate"] = usdjpy_close
        result["usdjpy_return_1d"] = usdjpy_returns
        result["usdjpy_return_5d"] = usdjpy_close.pct_change(5)
        result["usdjpy_return_20d"] = usdjpy_close.pct_change(20)

        # ローリング相関
        result["corr_usdjpy_20d"] = stock_returns.rolling(
            settings.CORRELATION_PERIOD
        ).corr(usdjpy_returns)

        # ドル円の移動平均乖離
        usdjpy_sma25 = usdjpy_close.rolling(25).mean()
        result["usdjpy_sma25_deviation"] = (
            (usdjpy_close - usdjpy_sma25) / usdjpy_sma25
        )

        # 円高/円安トレンド
        result["yen_trend_20d"] = usdjpy_close.pct_change(20)

    # ============================================================
    # 市場全体の状態指標
    # ============================================================
    if "nikkei225" in indices:
        nikkei_close = _align_index(indices["nikkei225"]["Close"], result.index)

        # 市場のボラティリティ
        nikkei_vol = nikkei_close.pct_change().rolling(20).std() * np.sqrt(252)
        result["market_volatility_20d"] = nikkei_vol

        # 市場のモメンタム
        result["market_momentum_5d"] = nikkei_close.pct_change(5)
        result["market_momentum_20d"] = nikkei_close.pct_change(20)

    logger.debug(f"マーケット連動指標計算完了: {len(result.columns)} カラム")
    return result


def _align_index(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    指数のSeriesを個別銘柄のインデックスに合わせる。
    営業日の違いを前方補完で対応する。
    """
    # インデックスの名前を統一
    series = series.copy()
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)

    # リインデックスして前方補完
    aligned = series.reindex(target_index, method="ffill")
    return aligned
