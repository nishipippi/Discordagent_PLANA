import sqlite3
from pathlib import Path
import os

DATABASE_DIR = Path(__file__).parent.parent / "data"
DATABASE_PATH = DATABASE_DIR / "memory.db"

def initialize_db():
    """
    データベースディレクトリを作成し、データベースに接続してテーブルを初期化します。
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # 既存のテーブルを削除（開発時のみ推奨）
    cursor.execute("DROP TABLE IF EXISTS memories;")

    # memories テーブルの作成
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'generic',
            memory_value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(server_id, memory_key)
        );
    """)

    # server_id と memory_key の複合インデックスを作成
    # UNIQUE制約を追加したため、このインデックスは不要になる可能性が高いが、
    # 既存のデータベースとの互換性を考慮し、ここでは削除しないでおく。
    # ただし、もし問題が発生する場合は、この行を削除することを検討する。
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_server_key ON memories (server_id, memory_key);
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

def get_db_connection():
    """
    データベース接続を返します。
    """
    return sqlite3.connect(DATABASE_PATH)

if __name__ == "__main__":
    initialize_db()
