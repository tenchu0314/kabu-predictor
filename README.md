# Kabu Predictor 🎯

日本株の値動きを予測し、毎日のおすすめ株Top 10を出力する株価予測システム。

## 特徴

- 📊 **時価総額1000億円以上**の日本株を対象
- 🤖 **LightGBM** + Optunaによる機械学習予測
- 📈 テクニカル指標（40+）、ファンダメンタル、マーケット連動指標を活用
- 🔄 **ウォークフォワード検証**によるバックテスト
- 🎯 翌日/5日/20日/60日の**マルチホライゾン予測**
- 💎 予測スコア × ファンダメンタル × リスク調整の**総合スコアリング**
- 🧠 **Gemini API**による銘柄レビュー・コメント生成
- ⏰ cron対応の自動実行（毎朝6:00 JST）

## セットアップ

### 1. 仮想環境の作成

```bash
cd ~/kabu-predictor
python3 -m venv venv
source venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

```bash
# Gemini API Keyを設定
echo 'export GEMINI_API_KEY=your-api-key-here' > .env
source .env
```

## 使い方

### フルパイプライン実行（全フェーズ）

```bash
python main.py
```

### フェーズ別実行

```bash
# Phase 1: データ取得のみ
python main.py --phase data

# Phase 2: 特徴量生成のみ
python main.py --phase features

# Phase 3: モデル学習のみ
python main.py --phase train

# Phase 4: 予測・レポートのみ
python main.py --phase predict
```

### オプション

```bash
# 銘柄リストを強制更新
python main.py --update-stocks

# Optuna最適化をスキップ（高速実行）
python main.py --no-optimize
```

### cron設定

```bash
bash scripts/setup_cron.sh
```

## アーキテクチャ

```
main.py
├── Phase 1: データ取得
│   ├── JPX銘柄リスト → 時価総額フィルタ
│   ├── 株価データ（yfinance, 過去2年）
│   ├── 指数データ（日経225, ダウ, ドル円）
│   └── 財務データ（BS, PL, CF, 基本情報）
│
├── Phase 2: 特徴量生成
│   ├── テクニカル指標（SMA, MACD, RSI, BB, ATR...）
│   ├── ファンダメンタル指標（PER, PBR, ROE...）
│   └── マーケット連動指標（相関, 相対強度...）
│
├── Phase 3: モデル学習
│   ├── LightGBM × 4ホライゾン
│   ├── Optuna ハイパーパラメータ最適化
│   └── ウォークフォワード検証
│
└── Phase 4: 予測・レポート
    ├── 全銘柄予測 → 重み付きスコア
    ├── 総合スコア（予測 + ファンダ + リスク）
    ├── Top 10 ランキング
    ├── Gemini レビュー
    └── 日次レポート出力
```

## 予測ホライゾンと重み

| ホライゾン | 重み | 用途 |
|-----------|------|------|
| 翌営業日   | 30%  | デイトレード |
| 5営業日後  | 30%  | 数日保有 |
| 20営業日後 | 25%  | 中期トレンド |
| 60営業日後 | 15%  | 長期文脈 |

## 総合スコア構成

```
最終スコア = 0.50 × 予測スコア + 0.25 × ファンダメンタル + 0.25 × リスク調整
```

## ディレクトリ構成

```
kabu/
├── config/settings.py          # 設定
├── data/
│   ├── stocks/                 # 銘柄別データ
│   ├── indices/                # 指数データ
│   └── master/                 # 銘柄リスト
├── src/
│   ├── data_collector/         # データ取得
│   ├── feature_engineering/    # 特徴量計算
│   ├── model/                  # ML モデル
│   ├── scoring/                # スコアリング
│   └── utils/                  # ユーティリティ
├── models/                     # 学習済みモデル
├── outputs/daily_reports/      # 日次レポート
├── logs/                       # ログ
├── main.py                     # エントリポイント
└── scripts/setup_cron.sh       # cron設定
```

## 注意事項

- ⚠️ このツールは投資判断の補助として使用してください。投資の最終判断はご自身の責任で行ってください。
- 📡 yfinance はYahoo Financeの非公式APIであり、レートリミットの影響を受ける場合があります。
- 🕐 初回実行時はデータ取得に時間がかかります（300銘柄×2年分）。
- 💻 モデル学習（Optuna最適化あり）には数時間かかる場合があります。