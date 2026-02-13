"""
LightGBMモデルの学習
ウォークフォワード検証 + Optunaハイパーパラメータ最適化
"""
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optunaの冗長なログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_target_labels(
    df: pd.DataFrame,
    horizons: dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    予測ホライゾンに応じたターゲットラベル（0/1）を作成する。

    Parameters
    ----------
    df : pd.DataFrame
        Close カラムを持つ DataFrame
    horizons : dict
        {ホライゾン日数: 重み} の辞書

    Returns
    -------
    pd.DataFrame
        ターゲットラベルが追加された DataFrame
    """
    if horizons is None:
        horizons = settings.PREDICTION_HORIZONS

    result = df.copy()

    for days in horizons.keys():
        # N日後のリターン
        future_return = result["Close"].shift(-days) / result["Close"] - 1
        result[f"target_{days}d"] = (future_return > 0).astype(int)
        result[f"future_return_{days}d"] = future_return

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """特徴量カラムのリストを返す（ターゲット・メタデータカラムを除外）"""
    exclude_prefixes = ["target_", "future_return_"]
    exclude_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "day_of_week", "month",  # カテゴリカルとして別途処理
    ]

    feature_cols = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if col in exclude_cols:
            continue
        feature_cols.append(col)

    return feature_cols


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    学習データを準備する（NaN除去、特徴量/ターゲット分離）
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # ターゲットと特徴量を結合してNaN除去
    data = df[feature_cols + [target_col]].dropna()

    X = data[feature_cols]
    y = data[target_col]

    return X, y


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = settings.OPTUNA_N_TRIALS,
    timeout: int = settings.OPTUNA_TIMEOUT,
) -> dict:
    """
    OptunaでLightGBMのハイパーパラメータを最適化する。
    """
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": 1000,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "n_estimators": 1000,
    })

    logger.info(f"最適パラメータ (AUC={study.best_value:.4f}): {best_params}")
    return best_params


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    optimize: bool = True,
) -> lgb.Booster:
    """
    LightGBMモデルを学習する。
    """
    if params is None:
        if optimize:
            params = optimize_hyperparams(X_train, y_train, X_val, y_val)
        else:
            params = settings.LGBM_DEFAULT_PARAMS.copy()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    # n_estimatorsをnum_iterationsに変換
    num_iterations = params.pop("n_estimators", 1000)
    params.pop("early_stopping_rounds", None)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_iterations,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # 検証スコア算出
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred_binary),
        "precision": precision_score(y_val, y_pred_binary, zero_division=0),
        "recall": recall_score(y_val, y_pred_binary, zero_division=0),
        "f1": f1_score(y_val, y_pred_binary, zero_division=0),
        "log_loss": log_loss(y_val, y_pred),
    }

    logger.info(
        f"モデル学習完了 - AUC: {metrics['auc']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )

    return model


def walk_forward_train(
    all_data: dict[str, pd.DataFrame],
    target_horizon: int,
    feature_cols: list[str] | None = None,
    optimize: bool = True,
) -> tuple[lgb.Booster, dict]:
    """
    ウォークフォワード検証で最適モデルを学習する。

    全銘柄のデータを統合して学習する「ユニバース方式」を採用。
    各銘柄の特徴量を同一の特徴量空間に投影し、一つのモデルで全銘柄を予測する。

    Parameters
    ----------
    all_data : dict
        {ticker: DataFrame} 形式の全銘柄データ
    target_horizon : int
        予測ホライゾン（営業日数）
    feature_cols : list, optional
        特徴量カラム名のリスト
    optimize : bool
        Optunaでハイパーパラメータ最適化を行うか

    Returns
    -------
    tuple
        (最終モデル, 評価結果の辞書)
    """
    target_col = f"target_{target_horizon}d"

    # 全銘柄データを統合
    logger.info(f"ホライゾン {target_horizon}日: 全銘柄データ統合中...")
    combined_frames = []
    for ticker, df in all_data.items():
        if target_col not in df.columns:
            continue
        temp = df.copy()
        temp["ticker"] = ticker
        combined_frames.append(temp)

    if not combined_frames:
        raise ValueError(f"有効なデータがありません（target: {target_col}）")

    combined = pd.concat(combined_frames, axis=0)
    combined = combined.sort_index()

    if feature_cols is None:
        feature_cols = get_feature_columns(combined)
        # ticker カラムを除外
        feature_cols = [c for c in feature_cols if c != "ticker"]

    # NaN除去
    valid_data = combined[feature_cols + [target_col]].dropna()
    logger.info(f"有効サンプル数: {len(valid_data)}")

    # ウォークフォワード分割
    # 学習: 最初の期間, 検証: 直近期間
    dates = valid_data.index.unique().sort_values()
    total_days = len(dates)

    # 直近3ヶ月をテスト、その前1ヶ月を検証、残りを学習
    test_days = min(60, total_days // 5)
    val_days = min(20, total_days // 10)
    train_end_idx = total_days - test_days - val_days

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:train_end_idx + val_days]
    test_dates = dates[train_end_idx + val_days:]

    X_train = valid_data.loc[valid_data.index.isin(train_dates), feature_cols]
    y_train = valid_data.loc[valid_data.index.isin(train_dates), target_col]
    X_val = valid_data.loc[valid_data.index.isin(val_dates), feature_cols]
    y_val = valid_data.loc[valid_data.index.isin(val_dates), target_col]
    X_test = valid_data.loc[valid_data.index.isin(test_dates), feature_cols]
    y_test = valid_data.loc[valid_data.index.isin(test_dates), target_col]

    logger.info(
        f"データ分割 - 学習: {len(X_train)}, "
        f"検証: {len(X_val)}, テスト: {len(X_test)}"
    )

    # モデル学習
    model = train_model(X_train, y_train, X_val, y_val, optimize=optimize)

    # テストセットで評価
    y_test_pred = model.predict(X_test)
    y_test_binary = (y_test_pred > 0.5).astype(int)

    test_metrics = {
        "horizon": target_horizon,
        "auc": roc_auc_score(y_test, y_test_pred),
        "accuracy": accuracy_score(y_test, y_test_binary),
        "precision": precision_score(y_test, y_test_binary, zero_division=0),
        "recall": recall_score(y_test, y_test_binary, zero_division=0),
        "f1": f1_score(y_test, y_test_binary, zero_division=0),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
    }

    logger.info(
        f"テスト結果 ({target_horizon}日) - "
        f"AUC: {test_metrics['auc']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    # 特徴量重要度
    importance = model.feature_importance(importance_type="gain")
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    test_metrics["top_features"] = feature_importance[:20]

    logger.info("Top 10 重要特徴量:")
    for feat, imp in feature_importance[:10]:
        logger.info(f"  {feat}: {imp:.2f}")

    return model, test_metrics


def save_model(
    model: lgb.Booster,
    horizon: int,
    metrics: dict | None = None,
) -> Path:
    """学習済みモデルを保存する"""
    model_path = settings.MODEL_DIR / f"lgbm_horizon_{horizon}d.txt"
    model.save_model(str(model_path))

    if metrics:
        metrics_path = settings.MODEL_DIR / f"metrics_horizon_{horizon}d.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

    logger.info(f"モデル保存: {model_path}")
    return model_path


def load_model(horizon: int) -> Optional[lgb.Booster]:
    """保存済みモデルを読み込む"""
    model_path = settings.MODEL_DIR / f"lgbm_horizon_{horizon}d.txt"
    if not model_path.exists():
        logger.warning(f"モデルが見つかりません: {model_path}")
        return None

    model = lgb.Booster(model_file=str(model_path))
    logger.info(f"モデル読み込み: {model_path}")
    return model


def train_all_horizons(
    all_data: dict[str, pd.DataFrame],
    optimize: bool = True,
) -> dict[int, tuple[lgb.Booster, dict]]:
    """
    全ホライゾンのモデルを学習・保存する。
    """
    results = {}

    for horizon, weight in settings.PREDICTION_HORIZONS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ホライゾン {horizon}日 の学習開始 (重み: {weight})")
        logger.info(f"{'='*60}")

        try:
            model, metrics = walk_forward_train(
                all_data, horizon, optimize=optimize
            )
            save_model(model, horizon, metrics)
            results[horizon] = (model, metrics)
        except Exception as e:
            logger.error(f"ホライゾン {horizon}日 の学習失敗: {e}")

    return results
