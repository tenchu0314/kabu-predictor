#!/bin/bash
# Kabu Predictor - cron設定スクリプト
# 毎日 06:00 JST にパイプラインを実行する

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
    echo "    echo 'GEMINI_API_KEY=your-api-key' > ${ENV_FILE}"
    echo ""
fi

# cronジョブの内容
CRON_COMMAND="0 6 * * * cd ${PROJECT_DIR} && source ${ENV_FILE} 2>/dev/null; ${PYTHON} main.py >> ${PROJECT_DIR}/logs/cron.log 2>&1"

echo "以下のcronジョブを登録します:"
echo "  ${CRON_COMMAND}"
echo ""

read -p "登録しますか? (y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    # 既存のkabu-predictorジョブを削除して新しいものを追加
    (crontab -l 2>/dev/null | grep -v "kabu.*main.py"; echo "${CRON_COMMAND} # kabu-predictor") | crontab -
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
echo "  ${PYTHON} main.py"
echo ""
echo "フェーズ別実行:"
echo "  ${PYTHON} main.py --phase data      # データ取得のみ"
echo "  ${PYTHON} main.py --phase train     # 学習のみ"
echo "  ${PYTHON} main.py --phase predict   # 予測のみ"
echo "  ${PYTHON} main.py --update-stocks   # 銘柄リスト強制更新"
echo "================================================="
