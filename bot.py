# bot.py
# (メインのボットファイル - Discordクライアント、イベントハンドラ)

import discord
import os
import asyncio
from typing import Optional, Literal

# --- モジュールインポート ---
import config
import bot_constants
import llm_manager
import cache_manager
import search_handler
import command_handler
import discord_ui
from llm_provider import LLMProvider # 型ヒント用

# --- Discordクライアント設定 ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True # 投票機能やタイマーのメンションで必要になる可能性
intents.guilds = True
intents.reactions = True # 投票機能で必要

client = discord.Client(intents=intents)
discord_client_id: str = "Unknown" # on_readyで設定

# --- ステータスメッセージ更新関数 ---
async def update_presence():
    """DiscordのPresence（ステータス）を更新する"""
    provider_name = llm_manager.get_current_provider_name()
    provider_display_name = provider_name.capitalize() if provider_name else "N/A"

    # コマンドリスト (configから取る方が良いかも？)
    commands = "!gemini, !mistral, !timer, !poll, !csum, !cclear, !his | @メンション+!src/!dsrc <検索語>" # 検索コマンドを修正
    activity_text = f"命令待機中 ({provider_display_name}) | {commands}"

    # 機能制限のチェック
    lowload_available = bool(llm_manager.get_active_model_name('lowload'))
    primary_available = bool(llm_manager.get_active_model_name('primary'))
    search_available = bool(config.BRAVE_SEARCH_API_KEY)

    limited_features = []
    # llm_handler が None の場合はモデル利用不可
    if not llm_manager.get_current_provider():
        limited_features.extend(["Primaryモデル", "低負荷モデル"])
    else:
        if not primary_available: limited_features.append("Primaryモデル")
        if not lowload_available: limited_features.append("低負荷モデル")

    if not search_available: limited_features.append("検索")

    if limited_features:
         activity_text += f" (機能制限: {', '.join(limited_features)})"

    try:
        await client.change_presence(activity=discord.Game(name=activity_text))
        print(f"Presence updated: {activity_text}")
    except Exception as e:
        print(f"Error updating presence: {e}")


# --- イベントハンドラ ---
@client.event
async def on_ready():
    """ボット起動時の処理"""
    global discord_client_id

    if client.user:
        discord_client_id = str(client.user.id)
        print(f"Client ID 設定: {discord_client_id}")
        # ペルソナ指示にClient IDを反映
        llm_manager.set_persona_instruction(
            bot_constants.PERSONA_TEMPLATE.format(client_id=discord_client_id)
        )
    else:
        print("CRITICAL: Botユーザー情報の取得に失敗しました。")
        await client.close(); return

    if not config.DISCORD_TOKEN:
        print("CRITICAL: DISCORD_TOKEN が .env に未設定。")
        await client.close(); return

    if not config.BRAVE_SEARCH_API_KEY:
        print("WARNING: BRAVE_SEARCH_API_KEY が .env に未設定です。検索機能(!src, !dsrc)は利用できません。")

    # --- 初期LLMプロバイダーの初期化 ---
    print(f"Attempting to initialize initial provider: {config.INITIAL_LLM_PROVIDER_NAME}")
    initial_handler = await llm_manager.initialize_provider(config.INITIAL_LLM_PROVIDER_NAME)

    if not initial_handler:
        print(f"CRITICAL: Failed to initialize initial provider {config.INITIAL_LLM_PROVIDER_NAME}.")
        # フォールバック試行
        fallback_provider_name = 'MISTRAL' if config.INITIAL_LLM_PROVIDER_NAME == 'GEMINI' else 'GEMINI'
        print(f"Attempting to initialize fallback provider: {fallback_provider_name}")
        fallback_handler = await llm_manager.initialize_provider(fallback_provider_name)
        if not fallback_handler:
            print("CRITICAL: Failed to initialize any LLM provider. Bot cannot function properly.")
            # llm_manager内部の _llm_handler は None のまま

    # --- キャッシュディレクトリ作成 ---
    cache_manager.ensure_cache_directories()

    # --- 起動情報表示 ---
    print('--------------------------------------------------')
    print("Discord接続完了")
    if client.user: print(f'アカウント {client.user} ({discord_client_id}) としてログイン。')
    current_provider = llm_manager.get_current_provider()
    provider_name = llm_manager.get_current_provider_name()
    print(f'現在のプロバイダー: {provider_name}')

    if current_provider:
        primary_name = current_provider.get_model_name('primary') or "N/A"
        secondary_name = current_provider.get_model_name('secondary') or "N/A"
        lowload_name = current_provider.get_model_name('lowload') or "N/A"
        print(f'モデル設定: Primary={primary_name}, Secondary={secondary_name}, Lowload={lowload_name}')
        # OpenAI互換プロバイダーの場合、Base URLも表示
        if hasattr(current_provider, 'base_url') and getattr(current_provider, 'base_url'):
             print(f'API Base URL: {getattr(current_provider, "base_url")}')
    else:
        print("警告: LLMハンドラーが初期化されていません！ 機能しません。")

    if config.BRAVE_SEARCH_API_KEY:
        print("Brave Search API Key is set. Search features enabled.")
    else:
        print("Brave Search API Key is NOT set. Search features disabled.")
    print('--------------------------------------------------')

    # --- ステータスメッセージ設定 ---
    await update_presence()

    print("Bot is ready!")


@client.event
async def on_message(message: discord.Message):
    """メッセージ受信時の処理"""
    # --- 基本チェック ---
    if message.author == client.user: return # 自分自身のメッセージ
    if not message.guild: return # DM無視
    if not client.user: return # クライアント情報がない (起動直後など)

    # --- LLMハンドラーが利用不可の場合は基本応答しない ---
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        # メンションされた場合のみエラー応答
        if client.user.mentioned_in(message):
            await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not available)", mention_author=False)
        return # LLMが使えないので以降の処理はスキップ

    # --- メッセージ内容に基づく処理分岐 ---
    content_lower = message.content.lower().strip() if message.content else ""
    is_mention = client.user.mentioned_in(message)

    # 1. メンション付き検索コマンド (!src, !dsrc)
    if is_mention:
        # メンションを除去したコンテンツでコマンドをチェック
        mention_strings = [f'<@!{client.user.id}>', f'<@{client.user.id}>']
        content_without_mention = message.content
        for mention in mention_strings:
            content_without_mention = content_without_mention.replace(mention, '').strip()
        content_without_mention_lower = content_without_mention.lower()

        search_command_type: Optional[Literal['src', 'dsrc']] = None
        query_text = ""
        if content_without_mention_lower.startswith('!src '):
            search_command_type = 'src'
            query_text = content_without_mention[len('!src '):].strip()
        elif content_without_mention_lower.startswith('!dsrc '):
            search_command_type = 'dsrc'
            query_text = content_without_mention[len('!dsrc '):].strip()

        if search_command_type and query_text:
            # 検索コマンド実行
            asyncio.create_task(search_handler.handle_search_command(message, search_command_type, query_text))
            return # 検索コマンド処理に任せる

    # 2. その他のコマンド (!gemini, !mistral, !csum, !cclear, !timer, !poll, !his [メンションなし])
    # 注意: 検索コマンドは上記で処理されるため、handle_commandからは削除する
    command_processed = await command_handler.handle_command(message)
    if command_processed:
        if content_lower in ['!gemini', '!mistral']:
             await update_presence() # プロバイダー切り替え後は必ず更新
        return # コマンドが処理されたら終了

    # 3. メンション応答 (検索コマンドではなかった場合)
    if is_mention:
        # 上記の検索コマンド処理で return されていなければ、通常のメンション応答
        await command_handler.handle_mention(message, client.user)
        return # メンション応答処理完了

    # 4. 上記以外 (通常のメッセージなど) は無視


# --- BOT起動 ---
if __name__ == "__main__":
    # 依存ライブラリチェック (簡易版)
    try:
        import dotenv
        import google.generativeai
        import openai
        import httpx
        import aiofiles
        print("Required libraries seem to be installed.")
    except ImportError as e:
        print(f"CRITICAL: 必要なライブラリが不足しています: {e}")
        print("実行前に `pip install -r requirements.txt` または必要なライブラリをインストールしてください。")
        exit(1)
    except Exception as e:
         print(f"CRITICAL: 依存ライブラリチェック中に予期せぬエラー: {e}")
         exit(1)

    # BOT実行
    try:
        print("BOT起動処理開始...")
        if not config.DISCORD_TOKEN:
            print("CRITICAL: DISCORD_TOKEN が見つかりません。 .env ファイルを確認してください。")
            exit(1)
        client.run(config.DISCORD_TOKEN)
    except discord.LoginFailure:
        print("CRITICAL: 不正なDiscordトークンです。 .env ファイルを確認してください。")
    except discord.PrivilegedIntentsRequired:
        print("CRITICAL: 必要な特権インテント（Message Contentなど）が無効になっています。Discord Developer PortalでBotの設定を確認してください。")
    except Exception as e:
        print(f"CRITICAL: BOT実行中に予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()