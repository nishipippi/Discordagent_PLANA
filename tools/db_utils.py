import sqlite3
import json
import os
from typing import List, Dict, Any
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
