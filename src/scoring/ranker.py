"""
総合スコアリングとランキング
予測スコア、ファンダメンタルスコア、リスク調整スコアを統合する。
"""
from typing import Optional

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

    # 総合スコア
    w = settings.SCORE_WEIGHTS
    result["composite_score"] = (
        w["prediction"] * result["weighted_score"] +
        w["fundamental"] * result["fundamental_score"] +
        w["risk_adjusted"] * result["risk_adjusted_score"]
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
    lines.append(f"\n{'='*70}")
    lines.append(f"  📊 おすすめ株 Top {len(top_df)} ({now.strftime('%Y-%m-%d')})")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'順位':>4} | {'コード':>6} | {'銘柄名':<16} | "
        f"{'総合':>5} | {'予測':>5} | {'ファンダ':>5} | {'リスク':>5}"
    )
    lines.append("-" * 70)

    for _, row in top_df.iterrows():
        rank = int(row.get("rank", 0))
        code = str(row.get("code", ""))
        name = str(row.get("name", ""))[:16]
        composite = row.get("composite_score", 0)
        prediction = row.get("weighted_score", 0)
        fundamental = row.get("fundamental_score", 0)
        risk = row.get("risk_adjusted_score", 0)

        lines.append(
            f"  {rank:>2}  | {code:>6} | {name:<16} | "
            f"{composite:.3f} | {prediction:.3f} | {fundamental:.3f} | {risk:.3f}"
        )

    lines.append(f"{'='*70}\n")
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
