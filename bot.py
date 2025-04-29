# bot.py
# (メインのボットファイル - Discordクライアント、イベントハンドラ)

import discord
import os
import asyncio
from typing import Optional, Literal
import re

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
    if message.author == client.user: return
    if not message.guild: return
    if not client.user: return

    llm_handler = llm_manager.get_current_provider()
    # LLMが利用不可でも検索要否判断はさせたいが、検索自体はLLMが必要
    # メンション応答もLLMが必要なので、ここで弾くのは維持する
    if not llm_handler:
        if client.user.mentioned_in(message):
            await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not available)", mention_author=False)
        return

    content_lower = message.content.lower().strip() if message.content else ""
    is_mention = client.user.mentioned_in(message)

    # 1. 通常コマンド (!gemini, !mistral, etc.)
    #    検索コマンド(!src, !dsrc)はメンション必須のためここでは処理しない
    if not is_mention:
        command_processed = await command_handler.handle_command(message)
        if command_processed:
            if content_lower in ['!gemini', '!mistral']:
                 await update_presence()
            return # コマンド処理完了なら終了

    # 2. メンション付きメッセージの処理
    if is_mention:
        # メンション文字列を除去
        mention_strings = [f'<@!{client.user.id}>', f'<@{client.user.id}>']
        content_without_mention = message.content
        for mention in mention_strings:
            content_without_mention = content_without_mention.replace(mention, '').strip()

        # 2a. -nosrc フラグチェック
        no_search_flag_match = re.search(r'\s-nosrc\b', content_without_mention, re.IGNORECASE)
        question_text = content_without_mention # デフォルト
        perform_search_assessment = True # 検索要否判断を行うか
        if no_search_flag_match:
            question_text = content_without_mention.replace(no_search_flag_match.group(0), '').strip()
            perform_search_assessment = False
            print("'-nosrc' flag detected. Skipping search assessment.")

        # 2b. 検索コマンド (!src, !dsrc) チェック
        #    注意: コマンドと -nosrc が同時に指定された場合の挙動は未定義だが、コマンドを優先する
        search_command_match = re.match(r'!(src|dsrc)\s+(.*)', content_without_mention, re.IGNORECASE | re.DOTALL)
        if search_command_match:
            search_command_type = search_command_match.group(1).lower()
            query_text_for_command = search_command_match.group(2).strip()
            if query_text_for_command:
                # search_handler に処理を移譲
                asyncio.create_task(search_handler.handle_search_command(message, search_command_type, query_text_for_command))
            else:
                await message.reply(f"!{search_command_type} の後に検索内容を指定してください。", mention_author=False)
            return # 検索コマンド処理後は終了

        # 2c. 検索コマンド以外、またはメンションのみの場合
        if not question_text.strip(): # メンションのみ（フラグやコマンド除去後に空になった場合も含む）
             # -nosrcがあってもメンションのみなら応答
             await message.reply("…呼びましたか？", mention_author=False)
             return

        # 2d. 検索要否判断と応答 (-nosrcフラグがない場合)
        if perform_search_assessment:
            print(f"Assessing search necessity for mention: '{question_text[:50]}...'")
            # search_handler に検索判断と、必要に応じた検索実行、または通常応答の呼び出しを依頼
            await search_handler.assess_and_respond_to_mention(message, question_text)
        else:
            # -nosrc フラグがあったので、検索せずに通常のメンション応答
            print("Calling handle_mention directly due to -nosrc flag.")
            await command_handler.handle_mention(message, client.user, question_text=question_text, perform_search=False)
        return


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