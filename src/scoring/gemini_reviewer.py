"""
Gemini API を使ったレビュー・コメント生成
Top N 銘柄の妥当性チェックと一言コメントを生成する。
"""
import json
from typing import Optional

import pandas as pd

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_gemini_model():
    """Gemini APIのモデルインスタンスを取得する"""
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY が設定されていません")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        return model
    except ImportError:
        logger.error("google-generativeai パッケージがインストールされていません")
        return None
    except Exception as e:
        logger.error(f"Gemini API 初期化エラー: {e}")
        return None


def build_review_prompt(top_df: pd.DataFrame) -> str:
    """Geminiに送信するプロンプトを構築する"""
    stock_info = []
    for _, row in top_df.iterrows():
        info = {
            "順位": int(row.get("rank", 0)),
            "コード": str(row.get("code", "")),
            "銘柄名": str(row.get("name", "")),
            "総合スコア": round(float(row.get("composite_score", 0)), 3),
            "予測スコア": round(float(row.get("weighted_score", 0)), 3),
            "ファンダメンタルスコア": round(float(row.get("fundamental_score", 0)), 3),
            "リスク調整スコア": round(float(row.get("risk_adjusted_score", 0)), 3),
        }

        # 各ホライゾンの予測確率
        for horizon in settings.PREDICTION_HORIZONS.keys():
            col = f"prob_{horizon}d"
            if col in row:
                info[f"{horizon}日後上昇確率"] = round(float(row[col]), 3)

        stock_info.append(info)

    prompt = f"""あなたは日本株の投資アドバイザーです。
以下は機械学習モデルによって算出された、本日のおすすめ株ランキングTop {len(top_df)} です。

## ランキングデータ
```json
{json.dumps(stock_info, ensure_ascii=False, indent=2)}
```

## お願いしたいこと

1. **ランキングの妥当性チェック**
   - このランキングに明らかな問題がないかチェックしてください
   - セクターの偏りがあれば指摘してください

2. **各銘柄への一言コメント**（30〜50文字程度）
   - その銘柄の現在の投資テーマや注目ポイントを簡潔に
   - デイトレード〜数日保有を想定した観点で

3. **注意すべきリスク要因**
   - 全体的な市場リスク
   - 特定銘柄のリスク（もしあれば）

4. **総合コメント**
   - 全体的な所感を2〜3文で

## 回答形式
以下のJSON形式で回答してください:
```json
{{
  "validity_check": "ランキングの妥当性に関するコメント",
  "sector_bias": "セクター偏りに関する指摘",
  "stock_comments": [
    {{"code": "コード", "name": "銘柄名", "comment": "一言コメント"}},
    ...
  ],
  "risk_factors": ["リスク1", "リスク2", ...],
  "overall_comment": "総合コメント"
}}
```
"""
    return prompt


def review_with_gemini(top_df: pd.DataFrame) -> str:
    """
    Gemini APIでTop N銘柄のレビューを生成する。

    Returns
    -------
    str
        レビュー結果の整形テキスト
    """
    model = _get_gemini_model()
    if model is None:
        return "（Gemini APIが利用できないため、レビューをスキップしました）"

    prompt = build_review_prompt(top_df)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text

        # JSONを抽出して整形
        review = _parse_gemini_response(raw_text)
        formatted = _format_review(review)

        logger.info("Geminiレビュー生成完了")
        return formatted

    except Exception as e:
        logger.error(f"Gemini API エラー: {e}")
        return f"（Gemini APIエラー: {e}）"


def _parse_gemini_response(raw_text: str) -> dict:
    """Geminiの応答からJSONを抽出する"""
    # JSON部分を抽出（```json ... ``` を除去）
    text = raw_text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSON解析に失敗した場合は生テキストを返す
        return {
            "overall_comment": raw_text,
            "stock_comments": [],
            "risk_factors": [],
            "validity_check": "",
            "sector_bias": "",
        }


def _format_review(review: dict) -> str:
    """レビュー結果を読みやすいテキストに整形する"""
    lines = []

    # 妥当性チェック
    validity = review.get("validity_check", "")
    if validity:
        lines.append(f"🔍 妥当性チェック: {validity}")

    # セクター偏り
    sector_bias = review.get("sector_bias", "")
    if sector_bias:
        lines.append(f"📊 セクター偏り: {sector_bias}")

    # 各銘柄コメント
    comments = review.get("stock_comments", [])
    if comments:
        lines.append("\n💬 銘柄コメント:")
        for item in comments:
            code = item.get("code", "")
            name = item.get("name", "")
            comment = item.get("comment", "")
            lines.append(f"  {code} {name}: {comment}")

    # リスク要因
    risks = review.get("risk_factors", [])
    if risks:
        lines.append("\n⚠️ リスク要因:")
        for risk in risks:
            lines.append(f"  • {risk}")

    # 総合コメント
    overall = review.get("overall_comment", "")
    if overall:
        lines.append(f"\n📝 総合: {overall}")

    return "\n".join(lines)
