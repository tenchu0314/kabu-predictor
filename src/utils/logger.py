"""
ロガー設定
"""
import logging
import sys
from config import settings


def setup_logger(name: str = "kabu_predictor") -> logging.Logger:
    """アプリケーション全体のロガーをセットアップする"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ
    file_handler = logging.FileHandler(
        settings.LOG_FILE, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(settings.LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "kabu_predictor") -> logging.Logger:
    """既存のロガーを取得する（なければ作成）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
