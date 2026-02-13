"""
バックテスト・モデル評価
ウォークフォワードでの累積リターン・勝率等を評価する。
"""
import numpy as np
import pandas as pd

from config import settings
from src.utils.logger import get_logger
from src.utils.helpers import save_json

logger = get_logger(__name__)


def evaluate_backtest(
    predictions: pd.DataFrame,
    actual_returns: pd.DataFrame,
    top_n: int = settings.TOP_N,
) -> dict:
    """
    バックテスト評価を行う。
    予測スコア上位N銘柄に等金額投資した場合のパフォーマンスを計算する。

    Parameters
    ----------
    predictions : pd.DataFrame
        日付ごとの予測結果（各銘柄のスコア）
    actual_returns : pd.DataFrame
        実際のリターン
    top_n : int
        ポートフォリオの銘柄数

    Returns
    -------
    dict
        バックテスト評価結果
    """
    results = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_return": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "daily_returns": [],
    }

    if predictions.empty or actual_returns.empty:
        return results

    daily_returns = []

    dates = predictions.index.unique().sort_values()

    for date in dates:
        day_preds = predictions.loc[date]
        if isinstance(day_preds, pd.Series):
            # 1銘柄のみの場合
            continue

        # スコア上位N銘柄を選択
        top_tickers = day_preds.nlargest(top_n, "weighted_score")["ticker"].tolist()

        # 実際のリターンを取得
        day_returns = []
        for ticker in top_tickers:
            if ticker in actual_returns.columns:
                ret = actual_returns.loc[date, ticker] if date in actual_returns.index else None
                if ret is not None and not np.isnan(ret):
                    day_returns.append(ret)
                    results["total_trades"] += 1
                    if ret > 0:
                        results["winning_trades"] += 1
                    else:
                        results["losing_trades"] += 1

        # 等金額投資のポートフォリオリターン
        if day_returns:
            portfolio_return = np.mean(day_returns)
            daily_returns.append(portfolio_return)

    if daily_returns:
        daily_returns = np.array(daily_returns)
        results["daily_returns"] = daily_returns.tolist()
        results["avg_return"] = float(np.mean(daily_returns))
        results["total_return"] = float(np.prod(1 + daily_returns) - 1)
        results["win_rate"] = (
            results["winning_trades"] / results["total_trades"]
            if results["total_trades"] > 0 else 0
        )

        # シャープレシオ（年率換算）
        if np.std(daily_returns) > 0:
            results["sharpe_ratio"] = float(
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            )

        # 最大ドローダウン
        cumulative = np.cumprod(1 + daily_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        results["max_drawdown"] = float(np.min(drawdown))

    return results


def calculate_risk_adjusted_score(
    df: pd.DataFrame,
    lookback: int = 60,
) -> float:
    """
    リスク調整後スコアを計算する。
    シャープレシオ的な発想で、リターン/リスク比を評価する。
    ボラティリティや急騰による過熱感も考慮する。

    Parameters
    ----------
    df : pd.DataFrame
        株価データ（Close カラムが必要）
    lookback : int
        評価期間（営業日）

    Returns
    -------
    float
        リスク調整スコア（0〜1に正規化）
    """
    if len(df) < lookback:
        return 0.5  # データ不足の場合は中立

    recent = df.tail(lookback)
    returns = recent["Close"].pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.5

    # シャープレシオ（年率換算）
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # ソルティノレシオ（下方リスクのみ考慮）
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
    else:
        sortino = sharpe * 1.5  # 下落がない場合はボーナス

    # 最大ドローダウン
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    max_drawdown = ((cumulative - peak) / peak).min()

    # 勝率
    win_rate = (returns > 0).mean()

    # ボラティリティペナルティ（年率30%超で減点開始）
    annual_vol = returns.std() * np.sqrt(252)
    vol_penalty = max(0, (annual_vol - 0.30) / 0.40)  # 30%〜70% → 0〜1
    vol_penalty = min(1, vol_penalty)

    # 各指標を正規化してから合成
    sharpe_score = max(0, min(1, (sharpe + 2) / 6))      # -2〜4 → 0〜1
    sortino_score = max(0, min(1, (sortino + 2) / 8))     # -2〜6 → 0〜1
    dd_score = max(0, min(1, 1 + max_drawdown / 0.3))     # -30%〜0% → 0〜1
    wr_score = max(0, min(1, (win_rate - 0.3) / 0.4))     # 30%〜70% → 0〜1

    risk_score = (
        0.30 * sharpe_score +
        0.20 * sortino_score +
        0.20 * dd_score +
        0.15 * wr_score -
        0.15 * vol_penalty       # ボラティリティが高すぎる銘柄を減点
    )

    return max(0, min(1, risk_score))


def generate_backtest_report(
    results: dict,
    horizon: int,
) -> str:
    """バックテスト結果のレポートテキストを生成する"""
    report = []
    report.append(f"\n{'='*50}")
    report.append(f"バックテスト結果 (ホライゾン: {horizon}日)")
    report.append(f"{'='*50}")
    report.append(f"総取引数:     {results.get('total_trades', 0)}")
    report.append(f"勝ち:         {results.get('winning_trades', 0)}")
    report.append(f"負け:         {results.get('losing_trades', 0)}")
    report.append(f"勝率:         {results.get('win_rate', 0):.2%}")
    report.append(f"平均リターン:  {results.get('avg_return', 0):.4%}")
    report.append(f"累積リターン:  {results.get('total_return', 0):.2%}")
    report.append(f"シャープレシオ: {results.get('sharpe_ratio', 0):.3f}")
    report.append(f"最大DD:       {results.get('max_drawdown', 0):.2%}")
    report.append(f"{'='*50}\n")

    return "\n".join(report)
