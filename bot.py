import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import call_llm
from langchain_core.messages import HumanMessage, AIMessage
from tools.discord_tools import get_discord_messages
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

    # 直近10件のメッセージを取得
    messages = await get_discord_messages(bot, channel_id, limit=10)
    
    return AgentState(
        user_input=state.user_input,
        chat_history=messages,
        channel_id=channel_id,
        thread_id=state.thread_id
    )

workflow.add_node("fetch_chat_history", fetch_chat_history)
workflow.add_node("call_llm", call_llm)

workflow.set_entry_point("fetch_chat_history")
workflow.add_edge("fetch_chat_history", "call_llm")
workflow.add_edge("call_llm", END)

app = workflow.compile()

# 会話の状態を保持する辞書 (簡易的な永続化)
conversation_states: Dict[int, AgentState] = {}

@bot.event
async def on_ready():
    print(f'Botとしてログインしました: {bot.user}')

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

        # 既存の会話状態をロード、なければ新規作成
        if channel_id not in conversation_states:
            conversation_states[channel_id] = AgentState(
                user_input="",
                chat_history=[],
                channel_id=channel_id,
                thread_id=thread_id
            )
        current_state: AgentState = conversation_states[channel_id] # 辞書から直接取得

        # 現在のユーザー入力をStateに設定
        current_state.user_input = user_input

        # LangGraphを実行
        raw_final_state = await app.ainvoke(current_state)
        final_state: AgentState = AgentState(**raw_final_state)

        # 更新されたStateを保存
        conversation_states[channel_id] = final_state

        # 最終的な応答を取得
        ai_response = final_state.chat_history[-1].content if final_state.chat_history else "応答できませんでした。"

        await message.channel.send(f'{message.author.mention} {ai_response}')

    await bot.process_commands(message)

# Botを実行
if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("DISCORD_TOKENが設定されていません。")
