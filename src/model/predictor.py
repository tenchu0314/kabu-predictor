"""
学習済みモデルを使った予測実行
"""
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import settings
from src.model.trainer import (
    get_feature_columns,
    load_model,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_single_stock(
    df: pd.DataFrame,
    models: dict[int, lgb.Booster] | None = None,
    feature_cols: list[str] | None = None,
) -> dict:
    """
    単一銘柄の予測を実行する。

    Parameters
    ----------
    df : pd.DataFrame
        特徴量が計算済みのデータフレーム
    models : dict, optional
        {ホライゾン: モデル} の辞書。Noneの場合は保存済みモデルを読み込む
    feature_cols : list, optional
        特徴量カラム名のリスト

    Returns
    -------
    dict
        予測結果の辞書
    """
    if models is None:
        models = {}
        for horizon in settings.PREDICTION_HORIZONS.keys():
            model = load_model(horizon)
            if model is not None:
                models[horizon] = model

    if not models:
        logger.error("有効なモデルがありません")
        return {}

    if feature_cols is None:
        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if c != "ticker"]

    # 最新の行（直近営業日）を使って予測
    latest = df.iloc[-1:]
    available_cols = [c for c in feature_cols if c in latest.columns]
    X = latest[available_cols]

    # NaNが多い場合は警告
    nan_ratio = X.isna().sum().sum() / X.size
    if nan_ratio > 0.3:
        logger.warning(f"特徴量のNaN比率が高い: {nan_ratio:.1%}")

    # NaNを0で埋める（LightGBMはNaN対応だが念のため）
    X = X.fillna(0)

    predictions = {}
    weighted_score = 0.0

    for horizon, weight in settings.PREDICTION_HORIZONS.items():
        if horizon not in models:
            continue

        model = models[horizon]

        # モデルの特徴量数に合わせる
        model_n_features = model.num_feature()
        if len(available_cols) != model_n_features:
            # 特徴量の不一致を処理
            logger.debug(
                f"特徴量数不一致: モデル={model_n_features}, データ={len(available_cols)}"
            )
            # モデルの特徴量名を取得して合わせる
            model_features = model.feature_name()
            matched_cols = [c for c in model_features if c in available_cols]
            missing_cols = [c for c in model_features if c not in available_cols]

            X_pred = X[matched_cols].copy()
            for col in missing_cols:
                X_pred[col] = 0
            X_pred = X_pred[model_features]
        else:
            X_pred = X

        prob = model.predict(X_pred)[0]
        predictions[f"prob_{horizon}d"] = float(prob)
        weighted_score += prob * weight

    predictions["weighted_score"] = weighted_score
    predictions["date"] = str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])

    return predictions


def predict_all_stocks(
    all_data: dict[str, pd.DataFrame],
    models: dict[int, lgb.Booster] | None = None,
) -> pd.DataFrame:
    """
    全銘柄の予測を実行する。

    Returns
    -------
    pd.DataFrame
        全銘柄の予測結果
    """
    if models is None:
        models = {}
        for horizon in settings.PREDICTION_HORIZONS.keys():
            model = load_model(horizon)
            if model is not None:
                models[horizon] = model

    if not models:
        logger.error("有効なモデルがありません")
        return pd.DataFrame()

    logger.info(f"全銘柄予測開始: {len(all_data)} 件")
    results = []

    for ticker, df in all_data.items():
        try:
            pred = predict_single_stock(df, models)
            if pred:
                pred["ticker"] = ticker
                results.append(pred)
        except Exception as e:
            logger.debug(f"{ticker}: 予測エラー: {e}")

    if not results:
        logger.error("有効な予測結果がありません")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("weighted_score", ascending=False)
    result_df = result_df.reset_index(drop=True)

    logger.info(f"予測完了: {len(result_df)} 件")
    return result_df
