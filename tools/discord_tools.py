from discord.ext import commands
import discord # discordモジュールをインポート
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

async def get_discord_messages(bot: commands.Bot, channel_id: int, limit: int = 10) -> List[BaseMessage]:
    """
    指定されたチャンネルからメッセージ履歴を取得し、LangChainのBaseMessage形式に変換する。
    """
    channel = bot.get_channel(channel_id)
    if not channel:
        print(f"チャンネルID {channel_id} が見つかりません。")
        return []

    # メッセージ履歴を持つチャンネルタイプか確認
    if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.DMChannel, discord.GroupChannel)):
        print(f"チャンネルID {channel_id} はメッセージ履歴を持たないチャンネルタイプです: {type(channel)}")
        return []

    messages = []
    async for msg in channel.history(limit=limit):
        # Bot自身のメンションは除外
        if msg.author == bot.user:
            continue
        
        if msg.author.bot:
            messages.append(AIMessage(content=msg.content))
        else:
            messages.append(HumanMessage(content=msg.content))
    
    return messages[::-1]
