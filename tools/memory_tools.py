import sqlite3
import json
from datetime import datetime
from langchain_core.tools import Tool # BaseToolではなくToolをインポート
from pydantic import BaseModel, Field
from typing import Optional, Type
from tools.db_utils import get_db_connection
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # LLMの型ヒント用

# --- プロンプト読み込みユーティリティ (toolsディレクトリ内なので、promptsディレクトリへのパスを調整) ---
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"

def load_prompt_file(file_path: Path) -> str:
    """指定されたパスのプロンプトファイルを読み込む"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading prompt from {file_path}: {e}")
        return ""

STRUCTURE_MEMORY_PROMPT_PATH = PROMPTS_DIR / "structure_memory_prompt.txt"
ANSWER_FROM_MEMORY_PROMPT_PATH = PROMPTS_DIR / "answer_from_memory_prompt.txt"

# プロンプトテンプレートをロード
STRUCTURE_MEMORY_TEMPLATE = load_prompt_file(STRUCTURE_MEMORY_PROMPT_PATH)
ANSWER_FROM_MEMORY_TEMPLATE = load_prompt_file(ANSWER_FROM_MEMORY_PROMPT_PATH)

# --- 記憶ツール (remember_information) の入力スキーマ ---
class RememberInput(BaseModel):
    memory_key: str = Field(description="記憶する情報のキー。例: '2025年度前期授業日程表', '山田太郎の連絡先'")
    content_to_remember: str = Field(description="記憶する具体的な内容。テキスト形式で提供され、必要に応じて構造化されます。")
    server_id: str = Field(description="DiscordサーバーのID")
    channel_id: str = Field(description="情報が記憶されたチャンネルのID")
    user_id: str = Field(description="情報を記憶させたユーザーのID")

# --- 記憶ツール (remember_information) の実体関数 ---
def remember_information_func(
    memory_key: str,
    content_to_remember: str,
    server_id: str,
    channel_id: str,
    user_id: str,
    llm: ChatGoogleGenerativeAI # llmはpartialでバインドされる
) -> str:
    """
    ユーザーから提供された情報を記憶します。特に、授業日程表、連絡先リスト、重要なメモなどを構造化して保存するのに適しています。
    記憶する際には、何についての情報かを示すキー（例：'2025年度前期授業日程表'）と、具体的な内容を渡してください。
    """

    try:
        # LLMを使ってcontent_to_rememberを構造化する
        structure_prompt = ChatPromptTemplate.from_template(STRUCTURE_MEMORY_TEMPLATE)
        chain = structure_prompt | llm # 渡されたLLMを使用
        
        # LLMに構造化を依頼
        structured_response = chain.invoke({"user_provided_text": content_to_remember, "memory_key": memory_key})
        
        # LLMの応答からJSON文字列を抽出
        parsed_content = str(structured_response.content).strip()

        # LLMが生成したJSONが有効か検証
        json.loads(parsed_content) # 無効なJSONであればここでエラーが発生

    except Exception as e:
        print(f"Error structuring content with LLM: {e}. Saving raw content.")
        parsed_content = json.dumps({"raw_content": content_to_remember}) # 構造化失敗時はそのまま保存

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # memory_typeの簡易的な判定（LLMの構造化結果から判断することも可能だが、一旦キーベース）
    memory_type = 'generic'
    if '授業日程表' in memory_key or 'スケジュール' in memory_key:
        memory_type = 'schedule'
    elif '連絡先' in memory_key:
        memory_type = 'contact_list'
    elif 'todo' in memory_key.lower() or 'to do' in memory_key.lower():
        memory_type = 'todo'
    elif 'メモ' in memory_key:
        memory_type = 'note'

    try:
        # 既存のキーがあれば更新、なければ挿入
        cursor.execute("""
            INSERT INTO memories (server_id, channel_id, user_id, memory_key, memory_type, memory_value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(server_id, memory_key) DO UPDATE SET
                memory_value = EXCLUDED.memory_value,
                memory_type = EXCLUDED.memory_type,
                updated_at = CURRENT_TIMESTAMP;
        """, (
            server_id,
            channel_id,
            user_id,
            memory_key,
            memory_type,
            parsed_content,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        return f"「{memory_key}」を記憶しました。"
    except sqlite3.Error as e:
        conn.rollback()
        return f"情報の記憶中にエラーが発生しました: {e}"
    finally:
        conn.close()

# --- 想起ツール (recall_information) の入力スキーマ ---
class RecallInput(BaseModel):
    query: str = Field(description="記憶した情報について質問する内容。例: '月曜日の1限は何？', '山田太郎さんの電話番号を教えて'")
    server_id: str = Field(description="DiscordサーバーのID")

# --- 想起ツール (recall_information) の実体関数 ---
def recall_information_func(
    query: str,
    server_id: str,
    llm: ChatGoogleGenerativeAI # llmはpartialでバインドされる
) -> str:
    """
    以前に記憶した情報について質問に答えます。例えば、『月曜日の1限は何？』や『山田太郎さんの電話番号を教えて』のように使います。
    何について知りたいかを具体的に質問してください。
    """

    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # server_idで絞り込み、全ての記憶を取得
        cursor.execute("SELECT memory_key, memory_value FROM memories WHERE server_id = ?", (server_id,))
        memories = cursor.fetchall()

        if not memories:
            return "そのサーバーに関する記憶はまだありません。"

        # 取得した記憶をLLMに渡し、ユーザーの質問に答えさせるためのプロンプトを生成
        formatted_memories = []
        for key, value in memories:
            try:
                parsed_value = json.loads(value)
                # raw_contentがあればそれを使う、なければ元のJSON文字列をそのまま表示
                if "raw_content" in parsed_value:
                    formatted_memories.append(f"キー: {key}, 内容: {parsed_value['raw_content']}")
                else:
                    formatted_memories.append(f"キー: {key}, 内容: {json.dumps(parsed_value, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                formatted_memories.append(f"キー: {key}, 内容: {value}")
        
        # LLMに回答を生成させる
        answer_prompt = ChatPromptTemplate.from_template(ANSWER_FROM_MEMORY_TEMPLATE)
        chain = answer_prompt | llm # 渡されたLLMを使用
        
        response = chain.invoke({
            "retrieved_memory_json": "\n".join(formatted_memories), # LLMに渡す記憶情報
            "user_query": query
        })
        
        return str(response.content)

    except sqlite3.Error as e:
        return f"情報の想起中にエラーが発生しました: {e}"
    finally:
        conn.close()
