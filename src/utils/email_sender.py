"""
メール送信ユーティリティ
日次予測結果をメールで通知する。

送信方法:
  1. EMAIL_TO のみ設定 → ローカルの sendmail コマンドを使用
  2. SMTP_SERVER 等も設定 → 外部SMTPサーバー経由で送信
"""
import shutil
import smtplib
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _send_via_sendmail(msg: MIMEMultipart) -> bool:
    """ローカルの sendmail コマンドで送信する"""
    sendmail_path = shutil.which("sendmail")
    if not sendmail_path:
        logger.error(
            "sendmail が見つかりません。"
            "sudo apt install mailutils または postfix をインストールしてください"
        )
        return False

    try:
        proc = subprocess.run(
            [sendmail_path, "-t", "-oi"],
            input=msg.as_string(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.error(f"sendmail エラー: {proc.stderr}")
            return False

        logger.info(f"メール送信完了 (sendmail): {settings.EMAIL_TO}")
        return True

    except Exception as e:
        logger.error(f"sendmail 送信エラー: {e}")
        return False


def _send_via_smtp(msg: MIMEMultipart) -> bool:
    """外部SMTPサーバー経由で送信する"""
    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"メール送信完了 (SMTP): {settings.EMAIL_TO}")
        return True

    except Exception as e:
        logger.error(f"SMTP 送信エラー: {e}")
        return False


def send_email(subject: str, body: str, html_body: str = "") -> bool:
    """
    メールを送信する。
    SMTP設定があればSMTP経由、なければローカルの sendmail を使用する。

    Parameters
    ----------
    subject : str
        メールの件名
    body : str
        メール本文（プレーンテキスト）
    html_body : str, optional
        HTML形式の本文（指定時はマルチパートで送信）

    Returns
    -------
    bool
        送信成功なら True
    """
    if not settings.EMAIL_TO:
        logger.debug("EMAIL_TO が未設定のため、メール送信をスキップ")
        return False

    # メッセージ作成
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = settings.EMAIL_TO

    # From: SMTP設定があればSMTPユーザー、なければ hostname ベース
    if settings.SMTP_USER:
        msg["From"] = settings.SMTP_USER
    else:
        msg["From"] = f"kabu-predictor@{_get_hostname()}"

    # プレーンテキスト
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # HTML（あれば）
    if html_body:
        msg.attach(MIMEText(html_body, "html", "utf-8"))

    # 送信方法を選択
    if settings.SMTP_SERVER and settings.SMTP_USER and settings.SMTP_PASSWORD:
        return _send_via_smtp(msg)
    else:
        return _send_via_sendmail(msg)


def _get_hostname() -> str:
    """ホスト名を取得する"""
    import socket
    try:
        return socket.getfqdn()
    except Exception:
        return "localhost"


def send_daily_report_email(
    ranking_text: str,
    gemini_review: str,
) -> bool:
    """
    日次予測結果をメールで送信する。

    Parameters
    ----------
    ranking_text : str
        ランキング表示テキスト
    gemini_review : str
        Geminiレビューテキスト
    """
    from src.utils.helpers import get_jst_now

    now = get_jst_now()
    date_str = now.strftime("%Y-%m-%d")
    subject = f"📊 Kabu Predictor 日次レポート ({date_str})"

    # プレーンテキスト
    body = f"""Kabu Predictor 日次レポート
日付: {date_str}

{ranking_text}

{gemini_review}

---
このメールは Kabu Predictor により自動送信されています。
"""

    # HTML
    ranking_html = ranking_text.replace("\n", "<br>").replace(" ", "&nbsp;")
    review_html = gemini_review.replace("\n", "<br>")

    html_body = f"""
<html>
<body style="font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 20px 30px; border-radius: 10px 10px 0 0;">
        <h1 style="margin: 0; font-size: 22px;">📊 Kabu Predictor</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">日次レポート {date_str}</p>
    </div>
    <div style="background: white; padding: 25px 30px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h2 style="color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; font-size: 18px;">🏆 おすすめ株ランキング</h2>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 6px; font-size: 13px; overflow-x: auto; line-height: 1.5;">{ranking_text}</pre>

        <h2 style="color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; font-size: 18px; margin-top: 25px;">🧠 Gemini レビュー</h2>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; font-size: 14px; line-height: 1.8;">{review_html}</div>

        <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;">
        <p style="color: #999; font-size: 12px; text-align: center;">
            ⚠️ 投資判断の補助情報です。最終判断はご自身の責任で行ってください。<br>
            Kabu Predictor により自動送信
        </p>
    </div>
</body>
</html>
"""

    return send_email(subject, body, html_body)
