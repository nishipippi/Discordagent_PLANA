import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import call_llm
from langchain_core.messages import HumanMessage, AIMessage
from tools.discord_tools import get_discord_messages
from tools.db_utils import init_db, load_chat_history, save_chat_history # 追加
from typing import Dict, Optional # Optionalを追加

# .envファイルから環境変数を読み込む
load_dotenv()

# Discord Botトークンを取得
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# インテントを設定
intents = discord.Intents.default()
intents.message_content = True  # メッセージ内容の読み取りを許可

# Botのインスタンスを作成
bot = commands.Bot(command_prefix='!', intents=intents)

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

workflow.add_node("fetch_chat_history", fetch_chat_history)
workflow.add_node("call_llm", call_llm)

workflow.set_entry_point("fetch_chat_history")
workflow.add_edge("fetch_chat_history", "call_llm")
workflow.add_edge("call_llm", END)

app = workflow.compile()

@bot.event
async def on_ready():
    print(f'Botとしてログインしました: {bot.user}')
    init_db() # データベースの初期化を追加

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
