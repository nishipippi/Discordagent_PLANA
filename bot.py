import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import call_llm, should_search_node, execute_search_node # 新しいノードをインポート
from langchain_core.messages import HumanMessage, AIMessage
from tools.discord_tools import get_discord_messages
from tools.db_utils import init_db, load_chat_history, save_chat_history # 追加
from typing import Dict, Optional, Literal # OptionalとLiteralを追加

from tools.vector_store_utils import VectorStoreManager # 新規追加
from llm_config import get_google_api_key # APIキー取得関数をインポート

# .envファイルから環境変数を読み込む
load_dotenv()

# Discord Botトークンを取得
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# インテントを設定
intents = discord.Intents.default()
intents.message_content = True  # メッセージ内容の読み取りを許可

# Botのインスタンスを作成
class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store_manager: Optional[VectorStoreManager] = None # ベクトルストアマネージャーを保持

    async def setup_hook(self):
        # データベースの初期化
        init_db() # tools.db_utils の init_db を呼び出す
        print("データベースの準備完了。")

        # ベクトルストアの初期化
        try:
            self.vector_store_manager = VectorStoreManager()
            print("ベクトルストアの準備完了。")
        except ValueError as e:
            print(f"ベクトルストアの初期化に失敗しました: {e}")
        except Exception as e:
            print(f"予期せぬエラーでベクトルストアの初期化に失敗: {e}")

bot = MyBot(command_prefix='!', intents=intents)

# LangGraphグラフの構築
workflow = StateGraph(AgentState)

# 新しいノード: メッセージ履歴の取得
async def fetch_chat_history(state: AgentState) -> AgentState:
    channel_id = state.channel_id
    # thread_id = state.thread_id # 必要に応じてスレッド対応

    # 直近のメッセージを取得 (例: 過去10件)
    new_messages = await get_discord_messages(bot, channel_id, limit=10)
    
    # 既存のチャット履歴に新しいメッセージを追加
    # 重複を避けるため、新しいメッセージが既存の履歴にないか確認するロジックを追加することも検討
    updated_chat_history = state.chat_history + new_messages
    
    # 履歴の長さを制限 (例: 最新の20件を保持)
    max_history_length = 20
    if len(updated_chat_history) > max_history_length:
        updated_chat_history = updated_chat_history[-max_history_length:]

    return AgentState(
        user_input=state.user_input,
        chat_history=updated_chat_history, # 更新された履歴を設定
        channel_id=channel_id,
        thread_id=state.thread_id
    )

workflow.add_node("fetch_chat_history", fetch_chat_history) # 既存
workflow.add_node("should_search", should_search_node)    # 新規
workflow.add_node("execute_search", execute_search_node)  # 新規
workflow.add_node("call_llm", call_llm)                   # 既存

workflow.set_entry_point("fetch_chat_history")
workflow.add_edge("fetch_chat_history", "should_search")

def select_next_node_after_should_search(state: AgentState) -> Literal["execute_search", "call_llm"]:
    if state.should_search_decision == "yes":
        return "execute_search"
    else:
        return "call_llm" # 検索不要なら直接LLMへ

workflow.add_conditional_edges(
    "should_search",
    select_next_node_after_should_search,
    {
        "execute_search": "execute_search",
        "call_llm": "call_llm",
    }
)
workflow.add_edge("execute_search", "call_llm")
workflow.add_edge("call_llm", END)

app = workflow.compile()

@bot.event
async def on_ready():
    print(f'Botとしてログインしました: {bot.user}')
    # init_db() は setup_hook に移動したため、ここでの呼び出しは不要

    if bot.vector_store_manager:
        print("VectorStoreManager は正常に初期化されています。")
    else:
        print("警告: VectorStoreManager が初期化されていません。記憶・想起機能が動作しない可能性があります。")

@bot.event
async def on_message(message):
    # Bot自身のメッセージは無視する
    if message.author == bot.user:
        return

    # Botへのメンションを検出
    if bot.user and bot.user in message.mentions:
        user_input = message.content.replace(f'<@{bot.user.id}>', '').strip()
        channel_id = message.channel.id
        thread_id = message.channel.id if isinstance(message.channel, discord.Thread) else None

        # 既存の会話状態をデータベースからロード、なければ新規作成
        loaded_chat_history = load_chat_history(channel_id)
        current_state = AgentState(
            user_input=user_input, # ユーザー入力を初期設定
            chat_history=loaded_chat_history,
            channel_id=channel_id,
            thread_id=thread_id
        )
        # ユーザーの入力をチャット履歴に追加
        current_state.chat_history.append(HumanMessage(content=user_input))

        # LangGraphを実行
        raw_final_state = await app.ainvoke(current_state)
        final_state: AgentState = AgentState(**raw_final_state)

        # 最終的な応答を取得
        ai_response_content = final_state.chat_history[-1].content if final_state.chat_history else "応答できませんでした。"
        # AIの応答をチャット履歴に追加
        final_state.chat_history.append(AIMessage(content=ai_response_content))

        # 更新されたStateをデータベースに保存
        save_chat_history(channel_id, final_state.chat_history)

        await message.channel.send(f'{message.author.mention} {ai_response_content}')

    await bot.process_commands(message)

# Botを実行
if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("DISCORD_TOKENが設定されていません。")
