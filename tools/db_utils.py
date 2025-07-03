import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Union # Union をインポート
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

def save_chat_history(channel_id: str, chat_history: List[BaseMessage]):
    """指定されたチャンネルのチャット履歴をデータベースに保存する。"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversation_history WHERE channel_id = ?", (str(channel_id),))

    for i, msg in enumerate(chat_history):
        message_type = ""
        if isinstance(msg, HumanMessage):
            message_type = "human"
        elif isinstance(msg, AIMessage):
            message_type = "ai"
        # 他のメッセージタイプ (SystemMessage, ToolMessageなど) を考慮する場合はここに追加

        content_to_save: str
        if isinstance(msg.content, (list, dict)): # content がリストまたは辞書の場合
            try:
                content_to_save = json.dumps(msg.content)
            except TypeError as e:
                # JSONシリアライズできないオブジェクトが含まれる場合のエラーハンドリング
                print(f"Warning: Could not serialize content to JSON for saving: {e}. Saving as string.")
                content_to_save = str(msg.content)
        elif isinstance(msg.content, str): # content が文字列の場合
            content_to_save = msg.content
        else: # その他の型の場合 (フォールバックとして文字列化)
            print(f"Warning: Unexpected content type ({type(msg.content)}) for saving. Saving as string.")
            content_to_save = str(msg.content)

        if message_type: # message_type が設定されていれば保存
            cursor.execute(
                "INSERT INTO conversation_history (channel_id, message_index, message_type, content) VALUES (?, ?, ?, ?)",
                (str(channel_id), i, message_type, content_to_save)
            )
        else:
            print(f"Warning: Unknown message type for message at index {i}. Skipping save.")

    conn.commit()
    conn.close()

def load_chat_history(channel_id: str) -> List[BaseMessage]:
    """指定されたチャンネルのチャット履歴をデータベースからロードする。"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT message_type, content FROM conversation_history WHERE channel_id = ? ORDER BY message_index",
        (str(channel_id),)
    )
    messages: List[BaseMessage] = []
    for row in cursor.fetchall():
        msg_type, db_content_str = row
        
        loaded_content: Union[str, List[Any], Dict[str, Any]]
        try:
            # 文字列がJSON形式（リストまたは辞書）であるか試みる
            if (db_content_str.startswith('[') and db_content_str.endswith(']')) or \
               (db_content_str.startswith('{') and db_content_str.endswith('}')):
                loaded_content = json.loads(db_content_str)
            else:
                loaded_content = db_content_str # JSONでなければそのまま文字列
        except json.JSONDecodeError:
            loaded_content = db_content_str
        except Exception as e:
            print(f"Warning: Error during content deserialization: {e}. Using raw string content.")
            loaded_content = db_content_str

        final_content_for_message: Union[str, List[Union[str, Dict[str, Any]]]]

        if isinstance(loaded_content, str):
            final_content_for_message = loaded_content
        elif isinstance(loaded_content, dict):
            # Langchain Message content の仕様に基づき、dict はリストでラップする
            final_content_for_message = [loaded_content]
        elif isinstance(loaded_content, list):
            # リスト内のすべての要素が str または Dict であることを保証する
            processed_list: List[Union[str, Dict[str, Any]]] = []
            for item_idx, item in enumerate(loaded_content):
                if isinstance(item, str):
                    processed_list.append(item)
                elif isinstance(item, dict):
                    # item はJSONからロードされた Dict[str, Any] または互換性のある型と仮定
                    processed_list.append(item)
                else:
                    # 他の型（int, float, bool, None など）は文字列に変換
                    print(f"Warning: Item at index {item_idx} in loaded list is not str or dict (type: {type(item)}). Converting to string.")
                    processed_list.append(str(item))
            final_content_for_message = processed_list
        else:
            # loaded_content が上記の try-except ブロックから正しく型付けされていれば、
            # このケースには到達しないはず (str, List[Any], Dict[str, Any])
            # ただし、念のためフォールバック処理
            print(f"Warning: Unexpected type for loaded_content ({type(loaded_content)}). Attempting to convert to string.")
            final_content_for_message = str(loaded_content)

        if msg_type == "human":
            messages.append(HumanMessage(content=final_content_for_message))
        elif msg_type == "ai":
            messages.append(AIMessage(content=final_content_for_message))
        # 他のメッセージタイプをロードする場合の処理をここに追加
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
