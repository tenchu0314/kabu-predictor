# Kabu Predictor 🎯

日本株の値動きを予測し、毎日のおすすめ株Top 10を出力する株価予測システム。

## 特徴

- 📊 **時価総額1000億円以上**の日本株を対象
- 🤖 **LightGBM** + Optunaによる機械学習予測
- 📈 テクニカル指標（40+）、ファンダメンタル、マーケット連動指標を活用
- 🔄 **ウォークフォワード検証**によるバックテスト
- 🎯 翌日/5日/20日/60日の**マルチホライゾン予測**
- ❤️ 予測スコア × ファンダメンタル × リスク調整の**総合スコアリング**
- 🧠 **Gemini API**による銘柄レビュー・コメント生成
- 📢 **Discord / メール通知**で結果を自動配信
- ⏰ cron対応の自動実行（日次予測 + 週次モデル学習）

## 推奨環境

本システムは **cron による自動実行** と **機械学習モデルの定期的な再学習** を前提としており、以下のような環境での常時運用を想定しています。

- 🖥️ **自宅サーバー / VPS / クラウドVM**（常時稼働する Linux マシン）
- 🐧 **OS**: Ubuntu 22.04 以降推奨（WSL2 でも動作可）
- 🐍 **Python**: 3.10 以降
- 💾 **メモリ**: 8GB 以上推奨（Optuna 最適化時に消費が増加します）
- 💽 **ストレージ**: 5GB 以上の空き容量（銘柄データ・学習済みモデル保存用）

> 💡 日次予測は軽量ですが、週次のモデル再学習（Optuna 最適化含む）には数時間かかる場合があります。十分なスペックのマシンを常時稼働させておくことを推奨します。

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
cp .env.example .env
```

`.env` を編集して、必要な API キーを設定してください。詳細は `.env.example` を参照してください。

```bash
source .env
```

## 使い方

### フルパイプライン実行（初回セットアップ時）

```bash
python main.py
```

### 日次予測（データ取得 → 特徴量 → 予測、モデル学習なし）

```bash
python main.py --phase daily
```

### 週次モデル学習（銘柄更新 → データ取得 → 特徴量 → 学習）

```bash
python main.py --phase weekly
```

### フェーズ別実行

```bash
python main.py --phase data       # データ取得のみ
python main.py --phase features    # 特徴量生成のみ
python main.py --phase train       # モデル学習のみ
python main.py --phase predict     # 予測・レポートのみ
```

### オプション

```bash
python main.py --update-stocks     # 銘柄リストを強制更新
python main.py --no-optimize       # Optuna最適化をスキップ（高速実行）
```

### cron 自動実行の設定

以下のスクリプトで、日次予測（月〜金 6:00 JST）と週次学習（日曜 0:00 JST）の 2 つの cron ジョブを登録します。

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
    ├── 日次レポート出力
    └── Discord / メール通知
```

## 通知設定

予測結果を Discord またはメールで自動配信できます。優先順位は **Discord → メール** で、どちらも未設定なら通知なしで動作します。

### Discord（推奨）

`.env` に以下を設定するだけで、指定チャンネルにランキングと Gemini レビューが届きます。

```bash
export DISCORD_BOT_TOKEN=your-bot-token
export DISCORD_CHANNEL_ID=your-channel-id
```

- **Bot Token**: [Discord Developer Portal](https://discord.com/developers/applications) → Bot → Token
- **Channel ID**: Discord の設定 → 詳細設定 → 開発者モード ON → チャンネルを右クリック → 「IDをコピー」

> 💡 Bot には対象チャンネルへの「メッセージを送信」権限が必要です。

### メール（オプション）

Discord を使わない場合のフォールバックとしてメール通知も利用できます。

```bash
# ローカル sendmail を使う場合（EMAIL_TO のみでOK）
export EMAIL_TO=your-email@example.com

# 外部 SMTP を使う場合（Gmail 等）
export SMTP_SERVER=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your-email@gmail.com
export SMTP_PASSWORD=your-app-password
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
kabu-predictor/
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
- 📡 yfinance は Yahoo Finance の非公式 API であり、レートリミットの影響を受ける場合があります。
- 🕐 初回実行時はデータ取得に時間がかかります（対象全銘柄 × 2 年分の株価 + 財務データ）。
- 💻 モデル学習（Optuna 最適化あり）には数時間かかる場合があります。
- 🔁 キャッシュ機能により、同日中の再実行時は API リクエストがスキップされます。