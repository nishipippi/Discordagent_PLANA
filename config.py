# config.py
# (設定値とプロンプトテンプレートの管理)

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# --- 環境変数の読み込み ---
try:
    load_dotenv()
    print(".env ファイル読み込み成功。")
except Exception as e:
    print(f"警告: .env ファイル読み込み中にエラー: {e}")

# --- Discord 設定 ---
DISCORD_TOKEN: Optional[str] = os.getenv('DISCORD_TOKEN')
HISTORY_LIMIT: int = 10 # Discordチャンネル履歴取得件数

# --- LLM プロバイダー設定 ---
INITIAL_LLM_PROVIDER_NAME: str = os.getenv('LLM_PROVIDER', 'GEMINI').upper()

# --- Gemini 設定 ---
GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_CONFIG: Dict[str, Optional[str]] = {
    'primary': os.getenv('GEMINI_PRIMARY_MODEL', 'gemini-1.5-pro-latest'),
    'secondary': os.getenv('GEMINI_SECONDARY_MODEL', 'gemini-1.5-flash-latest'),
    'lowload': os.getenv('GEMINI_LOWLOAD_MODEL', 'gemini-1.5-flash-latest')
}

# --- Mistral (OpenAI互換) 設定 ---
MISTRAL_API_KEY: Optional[str] = os.getenv('MISTRAL_API_KEY')
MISTRAL_API_BASE_URL: Optional[str] = os.getenv('MISTRAL_API_BASE_URL', 'https://api.mistral.ai/v1')
MISTRAL_MODEL_CONFIG: Dict[str, Optional[str]] = {
    'primary': os.getenv('MISTRAL_PRIMARY_MODEL', 'mistral-large-latest'),
    'secondary': os.getenv('MISTRAL_SECONDARY_MODEL'), # None許容、後でPrimaryからコピー
    'lowload': os.getenv('MISTRAL_LOWLOAD_MODEL', 'mistral-small-latest')
}
# Mistralモデル設定調整: Secondaryが設定されていない場合、Primaryを使用
if not MISTRAL_MODEL_CONFIG.get('secondary') and MISTRAL_MODEL_CONFIG.get('primary'):
    MISTRAL_MODEL_CONFIG['secondary'] = MISTRAL_MODEL_CONFIG.get('primary')

# --- Brave Search API 設定 ---
BRAVE_SEARCH_API_KEY: Optional[str] = os.getenv('BRAVE_SEARCH_API_KEY')
BRAVE_SEARCH_API_URL: str = "https://api.search.brave.com/res/v1/web/search"
BRAVE_API_DELAY: float = 1.1 # Brave API Rate Limit: 1 req/sec (Free plan) + safety margin

# --- 検索機能設定 ---
MAX_SEARCH_RESULTS: int = 5 # 検索結果から使用するページの数
MAX_CONTENT_LENGTH_PER_URL: int = 10000 # 各URLから抽出する最大文字数
MAX_TOTAL_SEARCH_CONTENT_LENGTH: int = 50000 # LLMに渡す検索コンテンツ合計の最大文字数
DEEP_SEARCH_MAX_ITERATIONS: int = 3 # !dsrc の最大検索回数を調整 (API負荷考慮)
SEARCH_MIN_CONTENT_LENGTH: int = 50 # URL抽出時の最小文字数閾値を調整

# --- キャッシュ設定 ---
CACHE_DIR: str = "cache"
CACHE_LIMIT: int = 10 # 往復数 (保存されるエントリ数は *2)

# --- Deep Cache 設定 ---
DEEP_CACHE_DIR: str = "deep_cache"

# --- ボタン生成用設定 ---
MAX_FOLLOW_UP_BUTTONS: int = 3
FOLLOW_UP_BUTTON_TIMEOUT: float = 1800.0 # 秒 (30分)

# --- プロンプトテンプレート ---

# 検索クエリ生成プロンプトテンプレート
SEARCH_QUERY_GENERATION_PROMPT: str = """
以下の質問に答えるための情報を検索するための、検索クエリを生成してください。
最も関連性が高いと思われる検索クエリを、1行に1つ、最大3つまで出力してください。
検索クエリのみを出力し、それ以外のテキスト（例えば「はい、検索クエリは以下の通りです」のような前置き）は含めないでください。

Question: {question}
"""

# 検索クエリ生成プロンプトテンプレート (履歴あり - 修正)
SEARCH_QUERY_GENERATION_PROMPT_WITH_HISTORY: str = """
以下の元の質問に答えるための情報を検索する必要があります。

これまでに以下の検索クエリを試しましたが、十分な情報は見つかりませんでした:
{used_queries}

前回の検索結果では、以下の情報が不足していると判断されました:
{missing_info}

これらの試行と**不足情報**を踏まえ、まだ見つかっていない情報を得るために、**新しい**関連性の高い検索クエリを生成してください。
最も関連性が高いと思われる新しい検索クエリを、1行に1つ、最大3つまで出力してください。
検索クエリのみを出力し、それ以外のテキストは含めないでください。

Original Question: {question}

New Search Queries:
"""

# 最終応答生成プロンプトテンプレート
SEARCH_ANSWER_PROMPT: str = """
あなたはAIアシスタントの「プラナ」です。以下の検索結果と元の質問に基づき、ユーザーへの包括的なレポートを作成してください。
検索結果の情報を組み合わせ、単なるコピーペーストではなく、情報を統合して分かりやすく説明してください。プラナの性格（簡潔、やや無口、少し毒舌だが親切）を反映させてください。

最後に、回答の根拠となった情報源のURLを以下の形式でリストアップしてください。このリストは必ず含めてください。
```markdown
**参照ソース:**
- <URL1>
- <URL2>
...
```

検索結果に必要な情報が含まれていない場合は、提供された検索結果からは完全な回答が見つけられなかったと正直に述べてください。その際も参照したURLがあればリストアップしてください。

Search Results:
---
{search_results_text}
---

Original Question: {question}

"""

# dsrc用検索結果評価プロンプトテンプレート
DEEP_SEARCH_ASSESSMENT_PROMPT: str = """
以下の元の質問と、現在までに収集された検索結果を分析してください。
検索結果の情報が、元の質問に完全に答えるために十分かどうかを判断してください。

- **情報が十分な場合:** 「COMPLETE」という単語のみを出力してください。
- **情報が不十分な場合:** 「INCOMPLETE: 」に続けて、まだ不足している情報の種類や、次に追加で検索すべきキーワードや質問を簡潔に記述してください。

応答には上記以外の情報を含めないでください。

Original Question: {question}

Search Results:
---
{search_results_text}
---

Assessment:
"""

# Deep Cache抽出プロンプト
DEEP_CACHE_EXTRACT_PROMPT: str = """
以下の会話履歴から、今後の会話の文脈として重要となりそうな事実、キーポイント、設定、ユーザーの好みなどを簡潔に抽出してください。箇条書きで記述してください。抽出する情報がない場合は「抽出情報なし」とだけ出力してください。

--- 会話履歴 ---
{history_text}
--- ここまで ---

抽出結果:
"""

# Deep Cache統合プロンプト
DEEP_CACHE_MERGE_PROMPT: str = """
以下の「既存の要約」と「新しい情報」を統合し、重複を排除し、関連情報をまとめ、一つの簡潔な要約を作成してください。箇条書き形式を維持してください。

--- 既存の要約 ---
{existing_summary}
--- ここまで ---

--- 新しい情報 ---
{new_summary}
--- ここまで ---

統合後の要約:
"""

# Deep Cache整理プロンプト
DEEP_CACHE_SUMMARIZE_PROMPT: str = """
以下の長期記憶の要約をレビューし、古くなった情報、矛盾する情報、冗長な記述を整理・削除して、より簡潔で一貫性のある最新の要約に書き直してください。箇条書き形式を維持してください。

--- 整理対象の要約 ---
{summary_to_clean}
--- ここまで ---

整理後の要約:
"""

# 追跡質問生成プロンプト
FOLLOW_UP_PROMPT: str = """
以下の直近の会話履歴を踏まえ、ユーザーが次に関心を持ちそうな質問やアクションを最大{max_buttons}つ提案してください。それぞれの提案は、Discordのボタンラベルとして表示される15文字程度の短いテキストにしてください。提案が不要な場合や、適切な提案が思いつかない場合は「提案なし」とだけ出力してください。提案は簡潔かつ具体的にしてください。

--- 直近の会話履歴 ---
{recent_history_text}
--- ここまで ---

提案（各行に1つずつ記述、最大{max_buttons}行）:
"""