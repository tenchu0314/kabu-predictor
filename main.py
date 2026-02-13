#!/usr/bin/env python3
"""
Kabu Predictor - メインパイプライン
全フェーズを順番に実行する。

Usage:
    python main.py                  # フルパイプライン実行（全フェーズ）
    python main.py --phase daily    # 日次実行（データ取得+予測、モデル学習なし）
    python main.py --phase weekly   # 週次実行（銘柄更新+データ取得+モデル学習）
    python main.py --phase data     # データ取得のみ
    python main.py --phase train    # モデル学習のみ
    python main.py --phase predict  # 予測・レポートのみ
    python main.py --update-stocks  # 銘柄リスト更新
    python main.py --no-optimize    # Optuna最適化をスキップ

スケジュール:
    日次 (毎朝 6:00 JST): python main.py --phase daily
    週次 (日曜 0:00 JST): python main.py --phase weekly
"""
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPATHに追加
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.utils.logger import setup_logger
from src.utils.helpers import get_jst_now

logger = setup_logger()


def phase_data(update_stocks: bool = False):
    """Phase 1: データ取得"""
    from src.data_collector.stock_list import (
        update_stock_list,
        get_target_stock_tickers,
        load_target_stocks,
    )
    from src.data_collector.price_fetcher import fetch_all_prices
    from src.data_collector.index_fetcher import fetch_all_indices
    from src.data_collector.financial_fetcher import fetch_all_financials

    logger.info("=" * 60)
    logger.info("Phase 1: データ取得")
    logger.info("=" * 60)

    # 銘柄リスト更新（指定時 or 日曜日）
    now = get_jst_now()
    if update_stocks or now.weekday() == settings.STOCK_LIST_UPDATE_DAY:
        logger.info("銘柄リスト更新中...")
        update_stock_list()
    else:
        target = load_target_stocks()
        if target is None or len(target) == 0:
            logger.info("銘柄リストが存在しないため新規作成します")
            update_stock_list()

    tickers = get_target_stock_tickers()
    logger.info(f"対象銘柄数: {len(tickers)}")

    # 株価データ取得
    logger.info("株価データ取得中...")
    fetch_all_prices(tickers)

    # 指数データ取得
    logger.info("指数データ取得中...")
    fetch_all_indices()

    # 財務データ取得
    logger.info("財務データ取得中...")
    fetch_all_financials(tickers)

    logger.info("Phase 1 完了")


def phase_features() -> dict:
    """Phase 2: 特徴量生成"""
    from src.data_collector.stock_list import get_target_stock_tickers
    from src.data_collector.price_fetcher import load_all_prices
    from src.data_collector.index_fetcher import load_all_indices
    from src.feature_engineering.technical import calculate_technical_features
    from src.feature_engineering.market import calculate_market_features
    from src.model.trainer import create_target_labels

    logger.info("=" * 60)
    logger.info("Phase 2: 特徴量生成")
    logger.info("=" * 60)

    # データ読み込み
    tickers = get_target_stock_tickers()
    price_data = load_all_prices(tickers)
    indices = load_all_indices()

    if not price_data:
        logger.error("株価データがありません。Phase 1を先に実行してください。")
        return {}

    logger.info(f"特徴量計算: {len(price_data)} 銘柄")

    all_data = {}
    success = 0
    fail = 0

    for ticker, df in price_data.items():
        try:
            # テクニカル指標
            featured_df = calculate_technical_features(df)

            # マーケット連動指標
            featured_df = calculate_market_features(featured_df, indices)

            # ターゲットラベル
            featured_df = create_target_labels(featured_df)

            all_data[ticker] = featured_df
            success += 1

        except Exception as e:
            logger.debug(f"{ticker}: 特徴量計算エラー: {e}")
            fail += 1

    logger.info(f"Phase 2 完了: 成功 {success}, 失敗 {fail}")
    return all_data


def phase_train(all_data: dict, optimize: bool = True):
    """Phase 3: モデル学習"""
    from src.model.trainer import train_all_horizons

    logger.info("=" * 60)
    logger.info("Phase 3: モデル学習")
    logger.info("=" * 60)

    if not all_data:
        logger.error("特徴量データがありません。Phase 2を先に実行してください。")
        return

    results = train_all_horizons(all_data, optimize=optimize)

    # 学習結果サマリ
    logger.info("\n=== 学習結果サマリ ===")
    for horizon, (model, metrics) in results.items():
        logger.info(
            f"  {horizon}日: AUC={metrics['auc']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )

    logger.info("Phase 3 完了")


def phase_predict(all_data: dict):
    """Phase 4: 予測・スコアリング・レポート"""
    from src.model.predictor import predict_all_stocks
    from src.scoring.ranker import (
        calculate_composite_score,
        get_top_n,
        save_daily_report,
        format_ranking_text,
    )
    from src.scoring.gemini_reviewer import review_with_gemini
    from src.data_collector.price_fetcher import load_all_prices

    logger.info("=" * 60)
    logger.info("Phase 4: 予測・スコアリング・レポート")
    logger.info("=" * 60)

    if not all_data:
        logger.error("特徴量データがありません。")
        return

    # 予測実行
    logger.info("予測実行中...")
    predictions = predict_all_stocks(all_data)

    if predictions.empty:
        logger.error("予測結果が空です")
        return

    # 総合スコア計算
    logger.info("総合スコア計算中...")
    price_data = load_all_prices()
    scored = calculate_composite_score(predictions, price_data)

    # Top N 抽出
    top_n = get_top_n(scored)

    # ランキング表示
    ranking_text = format_ranking_text(top_n)
    print(ranking_text)

    # Gemini レビュー
    logger.info("Geminiレビュー生成中...")
    gemini_review = review_with_gemini(top_n)
    print(gemini_review)

    # レポート保存
    save_daily_report(scored, top_n, gemini_review)

    # 通知送信（Discord優先、メールはフォールバック）
    from src.utils.discord_notifier import send_daily_report_discord
    from src.utils.email_sender import send_daily_report_email
    if not send_daily_report_discord(ranking_text, gemini_review):
        send_daily_report_email(ranking_text, gemini_review)

    logger.info("Phase 4 完了")


def run_daily():
    """
    日次実行: データ取得 → 特徴量生成 → 予測（既存モデル使用）
    毎朝 6:00 JST に cron で実行する想定。
    モデルの再学習は行わず、直近の学習済みモデルをそのまま使う。
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Kabu Predictor - 日次予測実行")
    logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        phase_data(update_stocks=False)
        all_data = phase_features()
        phase_predict(all_data)

        elapsed = datetime.now() - start_time
        logger.info(f"\n✅ 日次予測完了 (所要時間: {elapsed})")

    except Exception as e:
        logger.error(f"\n❌ 日次予測エラー: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_weekly(optimize: bool = True):
    """
    週次実行: 銘柄リスト更新 → データ取得 → 特徴量生成 → モデル再学習
    毎週日曜 0:00 JST に cron で実行する想定。
    予測レポートは出さず、モデルの更新のみ行う。
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Kabu Predictor - 週次モデル学習")
    logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        phase_data(update_stocks=True)
        all_data = phase_features()
        phase_train(all_data, optimize=optimize)

        elapsed = datetime.now() - start_time
        logger.info(f"\n✅ 週次学習完了 (所要時間: {elapsed})")

    except Exception as e:
        logger.error(f"\n❌ 週次学習エラー: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_full_pipeline(update_stocks: bool = False, optimize: bool = True):
    """フルパイプライン実行（全フェーズ）"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Kabu Predictor - フルパイプライン実行開始")
    logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # Phase 1: データ取得
        phase_data(update_stocks=update_stocks)

        # Phase 2: 特徴量生成
        all_data = phase_features()

        # Phase 3: モデル学習
        phase_train(all_data, optimize=optimize)

        # Phase 4: 予測・レポート
        phase_predict(all_data)

        elapsed = datetime.now() - start_time
        logger.info(f"\n✅ パイプライン完了 (所要時間: {elapsed})")

    except Exception as e:
        logger.error(f"\n❌ パイプラインエラー: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Kabu Predictor - 株価予測システム")
    parser.add_argument(
        "--phase",
        choices=["daily", "weekly", "data", "features", "train", "predict", "all"],
        default="all",
        help="実行するフェーズ (daily=日次予測, weekly=週次学習, all=全フェーズ)",
    )
    parser.add_argument(
        "--update-stocks",
        action="store_true",
        help="銘柄リストを強制更新",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Optunaハイパーパラメータ最適化をスキップ",
    )

    args = parser.parse_args()
    optimize = not args.no_optimize

    if args.phase == "daily":
        run_daily()
    elif args.phase == "weekly":
        run_weekly(optimize=optimize)
    elif args.phase == "all":
        run_full_pipeline(
            update_stocks=args.update_stocks,
            optimize=optimize,
        )
    elif args.phase == "data":
        phase_data(update_stocks=args.update_stocks)
    elif args.phase == "features":
        all_data = phase_features()
    elif args.phase == "train":
        all_data = phase_features()
        phase_train(all_data, optimize=optimize)
    elif args.phase == "predict":
        all_data = phase_features()
        phase_predict(all_data)


if __name__ == "__main__":
    main()
