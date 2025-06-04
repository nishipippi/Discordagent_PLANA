import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

DATABASE_PATH = "data/memory.db"

def init_db():
    """データベースを初期化し、必要なテーブルを作成する。"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            channel_id INTEGER NOT NULL,
            message_index INTEGER NOT NULL,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            PRIMARY KEY (channel_id, message_index)
        )
    """)
    # 構造化記憶テーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        server_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(server_id, channel_id, user_id, key)
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_key ON memories (user_id, key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_server_channel ON memories (server_id, channel_id)')
    conn.commit()
    conn.close()

def save_chat_history(channel_id: int, chat_history: List[BaseMessage]):
    """指定されたチャンネルのチャット履歴をデータベースに保存する。"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    # 既存の履歴を削除
    cursor.execute("DELETE FROM conversation_history WHERE channel_id = ?", (channel_id,))
    # 新しい履歴を挿入
    for i, msg in enumerate(chat_history):
        message_type = "human" if isinstance(msg, HumanMessage) else "ai"
        cursor.execute(
            "INSERT INTO conversation_history (channel_id, message_index, message_type, content) VALUES (?, ?, ?, ?)",
            (channel_id, i, message_type, msg.content)
        )
    conn.commit()
    conn.close()

def load_chat_history(channel_id: int) -> List[BaseMessage]:
    """指定されたチャンネルのチャット履歴をデータベースからロードする。"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT message_type, content FROM conversation_history WHERE channel_id = ? ORDER BY message_index",
        (channel_id,)
    )
    messages = []
    for row in cursor.fetchall():
        msg_type, content = row
        if msg_type == "human":
            messages.append(HumanMessage(content=content))
        elif msg_type == "ai":
            messages.append(AIMessage(content=content))
    conn.close()
    return messages

def save_memory(
    user_id: str,
    server_id: str,
    channel_id: str,
    original_text: str,
    structured_data: str # JSON文字列として受け取る
) -> Optional[int]:
    """
    構造化された記憶をデータベースに保存する。
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    memory_key: Optional[str] = None # 初期化
    try:
        # structured_data (JSON文字列) から key を抽出
        # 例: summary を key として使用
        structured_json = json.loads(structured_data)
        memory_key = structured_json.get("summary", original_text[:50] + "..." if len(original_text) > 50 else original_text)

        cursor.execute(
            """
            INSERT INTO memories (server_id, channel_id, user_id, key, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            (server_id, channel_id, user_id, memory_key, structured_data)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # UNIQUE制約違反の場合 (同じserver_id, channel_id, user_id, keyの組み合わせが既に存在)
        # memory_key が None の可能性があるのでチェック
        key_info = f"key '{memory_key}'" if memory_key else "an unknown key"
        print(f"Warning: Memory with {key_info} already exists for user {user_id} in {server_id}/{channel_id}. Skipping insertion.")
        return None
    except Exception as e:
        print(f"Error saving memory to DB: {e}")
        return None
    finally:
        conn.close()
