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

# --- LLM 入力制限 (文字数として扱う、簡易) ---
# Lowloadモデルに渡す検索結果の要約入力の最大文字数
MAX_INPUT_CHARS_FOR_SUMMARY: int = int(os.getenv('MAX_INPUT_CHARS_FOR_SUMMARY', '30000')) # デフォルト3万文字

# --- Brave Search API 設定 ---
BRAVE_SEARCH_API_KEY: Optional[str] = os.getenv('BRAVE_SEARCH_API_KEY')
BRAVE_SEARCH_API_URL: str = "https://api.search.brave.com/res/v1/web/search"
BRAVE_API_DELAY: float = 1.1 # Brave API Rate Limit: 1 req/sec (Free plan) + safety margin

# --- 検索機能設定 ---
MAX_SEARCH_RESULTS: int = 5 # 検索結果から使用するページの数
MAX_CONTENT_LENGTH_PER_URL: int = 30000 # 各URLから抽出する最大文字数
MAX_TOTAL_SEARCH_CONTENT_LENGTH: int = 150000 # LLMに渡す検索コンテンツ合計の最大文字数 (評価用など)
# MAX_TOTAL_SEARCH_CONTENT_LENGTH は、最終レポートの入力とは別に、評価や要約元として使う場合の最大長
# 最終レポートの入力は、要約される場合はMAX_INPUT_CHARS_FOR_SUMMARY以下になる

# --- !dsrc 設定 ---
DSRC_MAX_PLAN_STEPS: int = 3 # 計画の最大ステップ数
DSRC_MAX_ITERATIONS_PER_STEP: int = 3 # 各ステップでの最大検索試行回数
SEARCH_MIN_CONTENT_LENGTH: int = 100 # URL抽出時の最小文字数閾値を少し厳しく

# --- キャッシュ設定 ---
CACHE_DIR: str = "cache"
CACHE_LIMIT: int = 10 # 往復数 (保存されるエントリ数は *2)

# --- Deep Cache 設定 ---
DEEP_CACHE_DIR: str = "deep_cache"

# --- ボタン生成用設定 ---
MAX_FOLLOW_UP_BUTTONS: int = 3
FOLLOW_UP_BUTTON_TIMEOUT: float = 1800.0 # 秒 (30分)

# --- プロンプトテンプレート ---

# (既存のPERSONA_TEMPLATE)

# 検索要否判断プロンプト
SEARCH_NECESSITY_ASSESSMENT_PROMPT: str = """
以下のユーザーからの質問に答えるために、外部情報の検索が必要ですか？
一般的な知識で答えられる場合、創造的な応答や対話が求められる場合、ユーザー自身の意見を求めている場合などは「不要」と答えてください。
特定の事実、最新情報、数値データ、専門知識、特定のイベントや製品に関する情報が必要な場合は「必要」とだけ答えてください。

質問: {question}
判断:
"""

# 検索クエリ生成プロンプトテンプレート (変更なし)
SEARCH_QUERY_GENERATION_PROMPT: str = """
以下の質問に答えるための情報を検索するための、検索クエリを生成してください。
最も関連性が高いと思われる検索クエリを、1行に1つ、最大3つまで出力してください。
検索クエリのみを出力し、それ以外のテキスト（例えば「はい、検索クエリは以下の通りです」のような前置き）は含めないでください。

Question: {question}
"""

# dsrc用 検索クエリ生成プロンプトテンプレート (履歴あり)
DSRC_STEP_QUERY_GENERATION_PROMPT: str = """
以下の元の質問と、現在の調査ステップの説明、そして**これまでのクエリでは目的を達成できなかった**こと、アセスメントを踏まえ、このステップの目的達成に必要な情報を得るための**新しい**検索クエリを生成してください。

Original Question: {question}
Current Step: {step_description}
Previous Queries for this Step: {used_queries_for_step}
Missing Info from Last Assessment: {missing_info}

最も関連性が高いと思われる新しい検索クエリを**これまでのクエリでは目的を達成できなかった**ことを踏まえ、1行に1つ、最大3つまで出力してください。
検索クエリのみを出力し、それ以外のテキストは含めないでください。

New Search Queries:
"""

# 最終応答生成プロンプトテンプレート (!src / 自動検索用)
SIMPLE_SEARCH_ANSWER_PROMPT: str = """
以下の検索結果と元の質問に基づき、ユーザーへの応答を作成してください。

最後に、回答の根拠となった情報源のURLがあれば、以下の形式でリストアップしてください。
```markdown
**参照ソース:**
- <URL1>
- <URL2>
...
```

検索結果に必要な情報が含まれていない場合は、提供された検索結果からは回答が見つけられなかったと正直に述べてください。

Search Results:
---
{search_results_text}
---

Original Question: {question}

Response:
"""

# dsrc計画生成プロンプト
DSRC_PLAN_GENERATION_PROMPT: str = """
以下のユーザーの質問に包括的に答えるために、必要な情報を収集するための調査計画を立ててください。
計画は最大{max_steps}つのステップに分け、各ステップで具体的に何を調査すべきかを記述してください。
ステップは番号付きリストで記述し、計画のみを出力してください。他のテキストは含めないでください。

ユーザーの質問: {question}

調査計画:
"""

# dsrcステップ評価プロンプト
DSRC_STEP_ASSESSMENT_PROMPT: str = """
以下の調査計画の特定のステップについて、収集された情報がそのステップの目的を達成するために十分かどうかを評価してください。

元の質問: {question}
調査計画のステップ: {step_description}
収集された情報:
---
{search_results_text}
---

評価:
- **情報が十分で、ステップの目的を達成した場合:** 「COMPLETE」とだけ出力してください。
- **情報が不十分な場合:** 「INCOMPLETE: 」に続けて、このステップを完了するために**追加で必要**な具体的な情報や、次に試すべき検索の方向性を簡潔に記述してください。

応答には上記以外の情報を含めないでください。

評価結果:
"""

# dsrc最終レポート生成プロンプト (要約された結果を使用する可能性あり)
DSRC_FINAL_REPORT_PROMPT: str = """
以下の情報に基づいて、ユーザーへの包括的なレポートを作成してください。

*   **元の質問:** {question}
*   **調査計画:**\n{plan}
*   **調査プロセスで収集された情報:**\n{all_results_text} <!-- ここは生のURLごとのコンテンツ、またはその要約 -->
*   **各調査ステップの評価結果:**\n{all_assessments_text} <!-- 各ステップでの評価のまとめ -->

これらの情報を統合し、どのような情報が見つかったか、そして最終的に元の質問に対して何が言えるのかについて、レポートを作成してください。単なる情報の羅列ではなく、分析と統合を行ってください。

**調査プロセスで収集された情報**が要約されている場合は、その要約を参考に回答を生成してください。

最後に、回答の根拠となった情報源のURLを以下の形式でリストアップしてください。これは**調査プロセスで収集された情報**の元となった全URLから引用してください。要約された場合でも、元のURLリストは完全なものを提供します。
```markdown
**参照ソース:**
- <URL1>
- <URL2>
...
```

もし最終的に質問に完全には答えられなかった場合でも、調査プロセスと得られた情報を元に、現時点で分かっていることを正直に報告してください。

最終レポート:
"""

# 検索結果要約プロンプト (Lowload モデル用)
SUMMARIZE_SEARCH_RESULTS_PROMPT: str = """
以下の検索結果コンテンツを、元の質問に関連する重要な事実や情報を中心に、簡潔かつ包括的に要約してください。
要約は、箇条書きではなく、自然な文章形式で記述してください。
要約の目的は、この情報を使って元の質問に答えるためのキーポイントを提供することです。
元の質問に無関係で、ユーザーが求めていないであろう情報は無視してください。
もしコンテンツが元の質問に関連する重要な情報を含んでいない場合や、要約が難しい場合は、「要約できませんでした。」とだけ出力してください。

元の質問: {question}

検索結果コンテンツ:
---
{search_results_text}
---

要約:
"""


# (既存のDeep Cache系、追跡質問プロンプトはそのまま)
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