"""
総合スコアリングとランキング
予測スコア、ファンダメンタルスコア、リスク調整スコアを統合する。
過熱銘柄へのペナルティも適用する。
"""
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from src.data_collector.stock_list import load_target_stocks
from src.feature_engineering.fundamental import (
    calculate_fundamental_features,
    calculate_fundamental_score,
)
from src.model.evaluator import calculate_risk_adjusted_score
from src.utils.helpers import save_dataframe, save_json, get_jst_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_overheat_penalty(df: pd.DataFrame) -> float:
    """
    テクニカル指標に基づく過熱ペナルティを計算する。
    直近で急騰している銘柄ほどペナルティが大きくなる。

    判定指標:
      - RSI(14) が 75 超 → 過熱
      - 終値の 25日移動平均からの乖離率が +15% 超 → 過熱
      - 直近 5日リターンが +10% 超 → 短期急騰

    Returns
    -------
    float
        ペナルティ値（0 = ペナルティなし、最大 1.0）
    """
    if len(df) < 30:
        return 0.0

    close = df["Close"]
    recent_close = close.iloc[-1]

    penalties = []

    # 1. RSI(14) 過熱ペナルティ
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    if not np.isnan(current_rsi) and current_rsi > 75:
        # RSI 75〜100 → ペナルティ 0〜1
        penalties.append(min(1.0, (current_rsi - 75) / 25))

    # 2. 25日移動平均からの乖離率ペナルティ
    sma25 = close.rolling(25).mean().iloc[-1]
    if not np.isnan(sma25) and sma25 > 0:
        deviation = (recent_close - sma25) / sma25
        if deviation > 0.15:  # +15% 超で過熱
            penalties.append(min(1.0, (deviation - 0.15) / 0.25))

    # 3. 短期急騰ペナルティ（5日リターン）
    if len(close) >= 6:
        five_day_return = recent_close / close.iloc[-6] - 1
        if five_day_return > 0.10:  # +10% 超で過熱
            penalties.append(min(1.0, (five_day_return - 0.10) / 0.20))

    if not penalties:
        return 0.0

    # 最大のペナルティを採用（複数該当時は最も深刻な指標を重視）
    return max(penalties) * 0.7 + np.mean(penalties) * 0.3


def calculate_composite_score(
    prediction_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    各銘柄の総合スコアを計算する。

    Parameters
    ----------
    prediction_df : pd.DataFrame
        予測結果（weighted_score, ticker等）
    price_data : dict
        {ticker: DataFrame} 形式の株価データ

    Returns
    -------
    pd.DataFrame
        総合スコア付きのDataFrame
    """
    logger.info("総合スコア計算開始...")

    result = prediction_df.copy()

    # ファンダメンタルスコア
    fundamental_scores = []
    for _, row in result.iterrows():
        ticker = row["ticker"]
        fund_features = calculate_fundamental_features(ticker)
        fund_score = calculate_fundamental_score(fund_features)
        fundamental_scores.append(fund_score)

    result["fundamental_score"] = fundamental_scores

    # リスク調整スコア
    risk_scores = []
    for _, row in result.iterrows():
        ticker = row["ticker"]
        if ticker in price_data:
            risk_score = calculate_risk_adjusted_score(price_data[ticker])
        else:
            risk_score = 0.5
        risk_scores.append(risk_score)

    result["risk_adjusted_score"] = risk_scores

    # 過熱ペナルティ（急騰銘柄への偏りを抑制）
    overheat_penalties = []
    for _, row in result.iterrows():
        ticker = row["ticker"]
        if ticker in price_data:
            penalty = calculate_overheat_penalty(price_data[ticker])
        else:
            penalty = 0.0
        overheat_penalties.append(penalty)

    result["overheat_penalty"] = overheat_penalties

    # 総合スコア（過熱ペナルティを考慮）
    w = settings.SCORE_WEIGHTS
    base_score = (
        w["prediction"] * result["weighted_score"] +
        w["fundamental"] * result["fundamental_score"] +
        w["risk_adjusted"] * result["risk_adjusted_score"]
    )
    # 過熱ペナルティ: 最大30%減点
    penalty_factor = w.get("overheat_penalty", 0.30)
    result["composite_score"] = base_score * (
        1 - penalty_factor * result["overheat_penalty"]
    )

    # ランキング
    result = result.sort_values("composite_score", ascending=False)
    result = result.reset_index(drop=True)
    result["rank"] = result.index + 1

    # 銘柄名を追加
    target_stocks = load_target_stocks()
    if target_stocks is not None:
        name_map = dict(zip(target_stocks["ticker"], target_stocks["name"]))
        code_map = dict(zip(target_stocks["ticker"], target_stocks["code"]))
        result["name"] = result["ticker"].map(name_map).fillna("")
        result["code"] = result["ticker"].map(code_map).fillna("")

    logger.info(f"総合スコア計算完了: {len(result)} 件")

    return result


def get_top_n(
    scored_df: pd.DataFrame,
    n: int = settings.TOP_N,
) -> pd.DataFrame:
    """上位N銘柄を取得する"""
    return scored_df.head(n).copy()


def format_ranking_text(top_df: pd.DataFrame) -> str:
    """ランキングをテキスト形式で整形する"""
    now = get_jst_now()
    lines = []
    lines.append(f"\n{'='*76}")
    lines.append(f"  📊 おすすめ株 Top {len(top_df)} ({now.strftime('%Y-%m-%d')})")
    lines.append(f"{'='*76}")
    lines.append(
        f"{'順位':>4} | {'コード':>6} | {'銘柄名':<16} | "
        f"{'総合':>5} | {'予測':>5} | {'ファンダ':>5} | {'リスク':>5} | {'過熱':>4}"
    )
    lines.append("-" * 76)

    for _, row in top_df.iterrows():
        rank = int(row.get("rank", 0))
        code = str(row.get("code", ""))
        name = str(row.get("name", ""))[:16]
        composite = row.get("composite_score", 0)
        prediction = row.get("weighted_score", 0)
        fundamental = row.get("fundamental_score", 0)
        risk = row.get("risk_adjusted_score", 0)
        overheat = row.get("overheat_penalty", 0)

        # 過熱度をアイコンで表示
        if overheat >= 0.7:
            heat_icon = "🔥"
        elif overheat >= 0.3:
            heat_icon = "⚠️"
        else:
            heat_icon = "  "

        lines.append(
            f"  {rank:>2}  | {code:>6} | {name:<16} | "
            f"{composite:.3f} | {prediction:.3f} | {fundamental:.3f} | {risk:.3f} | {heat_icon}"
        )

    lines.append(f"{'='*76}")
    lines.append("  🔥 = 過熱注意  ⚠️ = やや過熱")
    lines.append("")
    return "\n".join(lines)


def save_daily_report(
    scored_df: pd.DataFrame,
    top_df: pd.DataFrame,
    gemini_review: str = "",
) -> None:
    """日次レポートを保存する"""
    now = get_jst_now()
    date_str = now.strftime("%Y-%m-%d")
    report_dir = settings.DAILY_REPORT_DIR / date_str
    report_dir.mkdir(parents=True, exist_ok=True)

    # 全銘柄スコア
    save_dataframe(scored_df, report_dir / "all_scores.csv", index=False)

    # Top N
    save_dataframe(top_df, report_dir / "top_picks.csv", index=False)

    # テキストレポート
    report_text = format_ranking_text(top_df)
    if gemini_review:
        report_text += f"\n\n📝 Gemini レビュー:\n{gemini_review}"

    report_path = report_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # JSON形式でも保存
    report_data = {
        "date": date_str,
        "top_picks": top_df.to_dict(orient="records"),
        "gemini_review": gemini_review,
        "summary": {
            "total_stocks_evaluated": len(scored_df),
            "top_n": len(top_df),
            "avg_composite_score": float(top_df["composite_score"].mean()),
        },
    }
    save_json(report_data, report_dir / "report.json")

    logger.info(f"日次レポート保存: {report_dir}")
    print(report_text)
