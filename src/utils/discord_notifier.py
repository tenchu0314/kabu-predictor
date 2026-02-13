"""
Discord 通知ユーティリティ
日次予測結果を Discord の指定チャンネルに送信する。
Discord Bot Token と Channel ID を .env に設定して使用する。
"""
import requests

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_MESSAGE_LENGTH = 2000  # Discord の1メッセージ上限


def _send_message(content: str) -> bool:
    """Discord チャンネルにテキストメッセージを送信する"""
    url = f"{DISCORD_API_BASE}/channels/{settings.DISCORD_CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {settings.DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"content": content}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Discord メッセージ送信エラー: {e}")
        return False


def _send_embed(title: str, description: str, color: int = 0x1A1A2E,
                fields: list[dict] | None = None) -> bool:
    """Discord チャンネルに Embed メッセージを送信する"""
    url = f"{DISCORD_API_BASE}/channels/{settings.DISCORD_CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {settings.DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    embed = {
        "title": title,
        "description": description[:4096],  # Embed description 上限
        "color": color,
    }
    if fields:
        embed["fields"] = fields[:25]  # Embed fields 上限

    payload = {"embeds": [embed]}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Discord Embed 送信エラー: {e}")
        return False


def _split_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """テキストを指定文字数以内で行単位に分割する"""
    lines = text.split("\n")
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks


def send_daily_report_discord(
    ranking_text: str,
    gemini_review: str,
) -> bool:
    """
    日次予測結果を Discord に送信する。

    Parameters
    ----------
    ranking_text : str
        ランキング表示テキスト
    gemini_review : str
        Geminiレビューテキスト

    Returns
    -------
    bool
        送信成功なら True
    """
    if not settings.DISCORD_BOT_TOKEN or not settings.DISCORD_CHANNEL_ID:
        logger.debug("Discord 設定が未設定のため、通知をスキップ")
        return False

    from src.utils.helpers import get_jst_now

    now = get_jst_now()
    date_str = now.strftime("%Y-%m-%d")

    success = True

    # 1. ヘッダー + ランキング（コードブロックで等幅表示）
    ranking_msg = f"# 📊 Kabu Predictor 日次レポート ({date_str})\n```\n{ranking_text}\n```"

    # ランキングが長い場合は分割
    if len(ranking_msg) <= MAX_MESSAGE_LENGTH:
        success &= _send_message(ranking_msg)
    else:
        success &= _send_message(f"# 📊 Kabu Predictor 日次レポート ({date_str})")
        for chunk in _split_text(f"```\n{ranking_text}\n```"):
            success &= _send_message(chunk)

    # 2. Gemini レビュー（Embedで見やすく）
    success &= _send_embed(
        title="🧠 Gemini レビュー",
        description=gemini_review,
        color=0xE94560,
    )

    # 3. フッター
    success &= _send_message(
        "-# ⚠️ 投資判断の補助情報です。最終判断はご自身の責任で行ってください。"
    )

    if success:
        logger.info("Discord 通知完了")
    else:
        logger.warning("Discord 通知で一部エラーが発生しました")

    return success
