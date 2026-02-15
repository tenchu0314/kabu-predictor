#!/bin/bash
# Kabu Predictor - cron設定スクリプト
# 日次: 毎朝 06:00 JST にデータ取得+予測（既存モデル使用）
# 週次: 毎週日曜 00:00 JST に銘柄更新+データ取得+モデル再学習

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python仮想環境のパス（必要に応じて変更）
PYTHON="${PROJECT_DIR}/venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python3)"
fi

# 環境変数設定ファイル
ENV_FILE="${PROJECT_DIR}/.env"

echo "================================================="
echo "Kabu Predictor - Cron Setup"
echo "================================================="
echo "Project Dir: ${PROJECT_DIR}"
echo "Python:      ${PYTHON}"
echo ""

# .envファイルの確認
if [ ! -f "$ENV_FILE" ]; then
    echo "⚠️  .env ファイルが見つかりません。作成してください:"
    echo "    echo 'export GEMINI_API_KEY=your-api-key' > ${ENV_FILE}"
    echo ""
fi

# cronジョブの内容
# 注意: cronはデフォルトで /bin/sh を使用するため、SHELL=/bin/bash を指定する
# . (dot) コマンドは POSIX 互換で source と同等
SHELL_LINE="SHELL=/bin/bash"
DAILY_CRON="0 6 * * 1-5 cd ${PROJECT_DIR} && . ${ENV_FILE} && ${PYTHON} main.py --phase daily >> ${PROJECT_DIR}/logs/cron_daily.log 2>&1 # kabu-daily"
WEEKLY_CRON="0 0 * * 0 cd ${PROJECT_DIR} && . ${ENV_FILE} && ${PYTHON} main.py --phase weekly >> ${PROJECT_DIR}/logs/cron_weekly.log 2>&1 # kabu-weekly"

echo "以下の2つのcronジョブを登録します:"
echo ""
echo "📊 日次予測 (月〜金 06:00):"
echo "  ${DAILY_CRON}"
echo ""
echo "🔧 週次学習 (日曜 00:00):"
echo "  ${WEEKLY_CRON}"
echo ""

read -p "登録しますか? (y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    # 既存のkabuジョブとSHELL設定を削除して新しいものを追加
    (crontab -l 2>/dev/null | grep -v "kabu-daily" | grep -v "kabu-weekly" | grep -v "^SHELL="; echo "${SHELL_LINE}"; echo "${DAILY_CRON}"; echo "${WEEKLY_CRON}") | crontab -
    echo ""
    echo "✅ cronジョブを登録しました"
    echo ""
    echo "現在のcronジョブ一覧:"
    crontab -l
else
    echo "キャンセルしました"
fi

echo ""
echo "================================================="
echo "手動実行する場合:"
echo "  cd ${PROJECT_DIR}"
echo "  source venv/bin/activate"
echo ""
echo "  # 日次予測（毎朝の処理）"
echo "  python main.py --phase daily"
echo ""
echo "  # 週次学習（モデル再学習）"
echo "  python main.py --phase weekly"
echo ""
echo "  # フルパイプライン（全フェーズ）"
echo "  python main.py"
echo ""
echo "  # 各フェーズ個別実行"
echo "  python main.py --phase data      # データ取得のみ"
echo "  python main.py --phase train     # 学習のみ"
echo "  python main.py --phase predict   # 予測のみ"
echo "  python main.py --update-stocks   # 銘柄リスト強制更新"
echo "================================================="
