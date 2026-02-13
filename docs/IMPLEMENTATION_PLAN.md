# Kabu Predictor 実装計画書

## 概要
日本株（時価総額1000億円以上）を対象とした株価予測システム。
主にデイトレード〜数日保有を想定し、毎朝6:00 JSTに自動実行。

## 確定仕様

### 予測ホライゾンと重み（デイトレ〜数日保有重視）
| ホライゾン | 重み | 備考 |
|-----------|------|------|
| 翌営業日 | 30% | メイン予測 |
| 5営業日後 | 30% | 数日保有向け |
| 20営業日後 | 25% | 中期トレンド確認 |
| 60営業日後 | 15% | 長期トレンド文脈 |

### 実行スケジュール
- 毎日 06:00 JST に cron 実行（米国市場終了後）
- モデル再学習: 毎日
- 銘柄リスト更新: 週1回（日曜日）

### データソース
- 株価データ: yfinance（過去2年分）
- 銘柄リスト: JPX上場銘柄一覧
- 指数: 日経225 (^N225), ダウ (^DJI), ドル円 (JPY=X)
- 財務データ: yfinance (balance_sheet, financials, cashflow)

### アルゴリズム
- Phase 1: LightGBM（ベースライン）
- Phase 2: バックテスト結果に基づき XGBoost, CatBoost 等を追加検討
- Phase 3: アンサンブル（成績の良いモデルを組み合わせ）

### スコアリング
```
最終スコア = 0.50 × 予測スコア + 0.25 × ファンダメンタルスコア + 0.25 × リスク調整スコア
```

### Gemini API 連携
- Top 10 銘柄のレビュー
- 各銘柄への一言コメント
- リスク注意事項

---

## フェーズ別実装計画

### Phase 1: 基盤構築
1. プロジェクト構成・設定ファイル
2. data_collector モジュール（銘柄リスト、株価、指数、財務）
3. データ保存・読み込みユーティリティ

### Phase 2: 特徴量エンジニアリング
4. テクニカル指標計算
5. ファンダメンタル指標計算
6. マーケット連動指標計算

### Phase 3: モデル構築
7. LightGBM モデルトレーナー
8. ウォークフォワード検証
9. バックテスト評価

### Phase 4: スコアリング・出力
10. 総合スコアリング・ランキング
11. Gemini API レビュー連携
12. レポート出力

### Phase 5: 自動化
13. メインパイプライン（main.py）
14. cron 設定スクリプト
15. ログ・エラーハンドリング

---

## ディレクトリ構成
```
kabu/
├── config/
│   └── settings.py
├── data/
│   ├── stocks/          # 銘柄別データ
│   ├── indices/         # 指数データ
│   └── master/          # 銘柄リスト
├── src/
│   ├── __init__.py
│   ├── data_collector/
│   │   ├── __init__.py
│   │   ├── stock_list.py
│   │   ├── price_fetcher.py
│   │   ├── financial_fetcher.py
│   │   └── index_fetcher.py
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── technical.py
│   │   ├── fundamental.py
│   │   └── market.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   └── evaluator.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── ranker.py
│   │   └── gemini_reviewer.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── models/              # 学習済みモデル保存
├── outputs/
│   └── daily_reports/
├── scripts/
│   └── setup_cron.sh
├── main.py
├── requirements.txt
├── README.md
└── docs/
    └── IMPLEMENTATION_PLAN.md
```
