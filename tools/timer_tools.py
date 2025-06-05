import asyncio
from datetime import datetime, timedelta
from typing import Type
import discord # 追加
from discord.ext import commands # 追加
from discord.abc import Messageable # 追加: Messageableをインポート

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class TimerInput(BaseModel):
    """Input for the timer tool."""
    minutes: int = Field(description="The number of minutes to set the timer for.")
    channel_id: str = Field(description="The ID of the Discord channel to send the notification to.")
    user_id: str = Field(description="The ID of the user to mention in the notification.")
    message: str = Field(description="The message to send when the timer goes off.")

async def _send_timer_notification(bot: commands.Bot, channel_id: str, user_id: str, message: str):
    """Sends a timer notification to the specified Discord channel."""
    try:
        channel = bot.get_channel(int(channel_id))
        if channel:
            # チャンネルがメッセージ送信可能か確認
            if isinstance(channel, Messageable):
                await channel.send(f"<@{user_id}> {message}")
                print(f"Timer notification sent to channel {channel_id} for user {user_id}.")
            else:
                print(f"Error: Channel {channel_id} (type: {type(channel).__name__}) is not a messageable channel.")
        else:
            print(f"Error: Channel {channel_id} not found for timer notification.")
    except Exception as e:
        print(f"Error sending timer notification: {e}")

async def _set_timer_func(bot: commands.Bot, minutes: int, channel_id: str, user_id: str, message: str) -> str:
    """Sets a timer and sends a notification to the specified Discord channel."""
    try:
        if minutes <= 0:
            return "Timer duration must be a positive number of minutes."

        # バックグラウンドで通知タスクを起動
        asyncio.create_task(_perform_timer_wait_and_notify(bot, minutes, channel_id, user_id, message))

        return f"タイマーを{minutes}分に設定しました。時間になったらお知らせします。" # すぐに確認メッセージを返す

    except Exception as e:
        return f"Error setting timer: {e}"

async def _perform_timer_wait_and_notify(bot: commands.Bot, minutes: int, channel_id: str, user_id: str, message: str):
    """Waits for the specified time and then sends the notification."""
    await asyncio.sleep(minutes * 60) # 秒に変換
    await _send_timer_notification(bot, channel_id, user_id, message)

def create_timer_tool(bot_instance: commands.Bot) -> StructuredTool:
    """Creates and returns the timer tool, binding the bot instance."""
    return StructuredTool.from_function(
        func=lambda minutes, channel_id, user_id, message: _set_timer_func(bot_instance, minutes, channel_id, user_id, message),
        name="set_timer",
        description="Sets a timer for a specified number of minutes and sends a notification message to a Discord channel, mentioning a user. The message will be sent after the specified time has elapsed.",
        args_schema=TimerInput,
        coroutine=lambda minutes, channel_id, user_id, message: _set_timer_func(bot_instance, minutes, channel_id, user_id, message),
    )
