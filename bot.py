# bot.py
# (メインのボットファイル - API呼び出しを抽象化)

import discord
import os
import asyncio
import re
import json
import aiofiles
import mimetypes
import base64
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# --- プロバイダーと定数をインポート ---
from llm_provider import LLMProvider # インターフェース
from gemini_provider import GeminiProvider # Gemini実装
from openai_compatible_provider import OpenAICompatibleProvider # OpenAI互換実装 (Mistral含む)
import bot_constants # 定数ファイル

# --- 設定項目 ---
# 0. 環境変数の読み込み
try:
    load_dotenv()
    print(".env ファイル読み込み成功。")
except Exception as e:
    print(f"警告: .env ファイル読み込み中にエラー: {e}")

LLM_PROVIDER_NAME = os.getenv('LLM_PROVIDER', 'GEMINI').upper()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_API_BASE_URL = os.getenv('MISTRAL_API_BASE_URL', 'https://api.mistral.ai/v1')

MODEL_CONFIG: Dict[str, str] = {}
API_KEY_FOR_PROVIDER: Optional[str] = None
API_BASE_URL_FOR_PROVIDER: Optional[str] = None

if LLM_PROVIDER_NAME == 'GEMINI':
    MODEL_CONFIG['primary'] = os.getenv('GEMINI_PRIMARY_MODEL', 'gemini-1.5-pro-latest')
    MODEL_CONFIG['secondary'] = os.getenv('GEMINI_SECONDARY_MODEL', 'gemini-1.5-flash-latest')
    MODEL_CONFIG['lowload'] = os.getenv('GEMINI_LOWLOAD_MODEL', 'gemini-1.5-flash-latest')
    API_KEY_FOR_PROVIDER = GEMINI_API_KEY
    API_BASE_URL_FOR_PROVIDER = None

elif LLM_PROVIDER_NAME == 'MISTRAL':
    MODEL_CONFIG['primary'] = os.getenv('MISTRAL_PRIMARY_MODEL', 'pixtral-large-latest')
    MODEL_CONFIG['secondary'] = os.getenv('MISTRAL_SECONDARY_MODEL', MODEL_CONFIG.get('primary', 'mistral-large-latest'))
    MODEL_CONFIG['lowload'] = os.getenv('MISTRAL_LOWLOAD_MODEL', 'mistral-small-latest')
    API_KEY_FOR_PROVIDER = MISTRAL_API_KEY
    API_BASE_URL_FOR_PROVIDER = MISTRAL_API_BASE_URL

else:
    print(f"Error: Unknown LLM_PROVIDER '{LLM_PROVIDER_NAME}' specified in .env. Please use 'GEMINI' or 'MISTRAL'.")
    print("Attempting to initialize with GEMINI provider settings as a fallback.")
    LLM_PROVIDER_NAME = 'GEMINI'
    MODEL_CONFIG['primary'] = os.getenv('GEMINI_PRIMARY_MODEL', 'gemini-1.5-pro-latest')
    MODEL_CONFIG['secondary'] = os.getenv('GEMINI_SECONDARY_MODEL', 'gemini-1.5-flash-latest')
    MODEL_CONFIG['lowload'] = os.getenv('GEMINI_LOWLOAD_MODEL', 'gemini-1.5-flash-latest')
    API_KEY_FOR_PROVIDER = GEMINI_API_KEY
    API_BASE_URL_FOR_PROVIDER = None


# 2. ペルソナ設定
PERSONA_TEMPLATE = bot_constants.PERSONA_TEMPLATE
PERSONA_INSTRUCTION = ""

# 3. 会話キャッシュ設定
CACHE_DIR = "cache"
CACHE_LIMIT = 10

# 4. Deep Cache 設定
DEEP_CACHE_DIR = "deep_cache"
DEEP_CACHE_EXTRACT_PROMPT = """
以下の過去の会話履歴から、ユーザーの好み、繰り返し話題になるトピック、重要な設定や決定事項、ユーザーに関する特筆すべき情報などを抽出し、箇条書きで簡潔にまとめてください。個人を特定しすぎる情報や一時的な挨拶などは除外してください。

--- 過去の会話履歴 ---
{history_text}
--- ここまで ---

抽出結果（箇条書き）:
"""
DEEP_CACHE_MERGE_PROMPT = """
以下の二つの箇条書きリスト（既存の長期記憶と、新しく抽出された情報）を統合し、重複する内容を賢く削除・整理して、一つの簡潔な箇条書きリストにまとめてください。ユーザーの好み、重要な決定事項、繰り返し話題になるトピックなどを中心に残し、情報の鮮度も考慮してください。リストが長くなりすぎる場合は、重要度の低いものから省略してください。

--- 既存の長期記憶 ---
{existing_summary}
--- ここまで ---

--- 新しく抽出された情報 ---
{new_summary}
--- ここまで ---

統合・整理後の長期記憶（箇条書きリスト）:
"""
DEEP_CACHE_SUMMARIZE_PROMPT = """
以下の長期記憶リストの内容を精査し、重複する項目を削除・統合し、より簡潔で分かりやすい形に整理してください。ユーザーの好み、重要な決定事項、繰り返し話題になるトピックなどを中心に残し、情報の鮮度も考慮してください。リストが長くなりすぎる場合は、重要度の低いものから省略してください。

--- 整理対象の長期記憶リスト ---
{summary_to_clean}
--- ここまで ---

整理後の長期記憶（箇条書きリスト）:
"""

# 5. Discordチャンネル履歴取得件数
HISTORY_LIMIT = 10

# 6. ボタン生成用設定
FOLLOW_UP_PROMPT = """
以下の直近の会話履歴を踏まえ、ユーザーが次に関心を持ちそうな質問やアクションを最大3つ提案してください。それぞれの提案は、Discordのボタンラベルとして表示される15文字程度の短いテキストにしてください。提案が不要な場合や、適切な提案が思いつかない場合は「提案なし」とだけ出力してください。提案は簡潔かつ具体的にしてください。

--- 直近の会話履歴 ---
{recent_history_text}
--- ここまで ---

提案（各行に1つずつ記述、最大3行）:
"""
MAX_FOLLOW_UP_BUTTONS = 3

# --- グローバル変数 ---
llm_handler: Optional[LLMProvider] = None
discord_client_id = "Unknown"

# --- 初期化処理 ---
async def initialize_llm_provider() -> bool:
    """設定に基づいてLLMプロバイダーを初期化する"""
    global llm_handler, PERSONA_INSTRUCTION
    if not discord_client_id or discord_client_id == "Unknown":
        print("警告: Discord Client ID が未設定です。PERSONA_INSTRUCTION が不完全になる可能性があります。")
    PERSONA_INSTRUCTION = PERSONA_TEMPLATE.format(client_id=discord_client_id)

    try:
        if LLM_PROVIDER_NAME == 'GEMINI':
            llm_handler = GeminiProvider()
            if not API_KEY_FOR_PROVIDER:
                 print("CRITICAL: API Key (GEMINI_API_KEY) not found in .env.")
                 llm_handler = None
                 return False

        elif LLM_PROVIDER_NAME == 'MISTRAL':
            llm_handler = OpenAICompatibleProvider()
            if not API_KEY_FOR_PROVIDER:
                 print("CRITICAL: API Key (MISTRAL_API_KEY) not found in .env.")
                 llm_handler = None
                 return False
            if not API_BASE_URL_FOR_PROVIDER:
                 print("CRITICAL: API Base URL (MISTRAL_API_BASE_URL) not found in .env.")
                 llm_handler = None
                 return False

        else:
            print(f"CRITICAL: Invalid LLM_PROVIDER '{LLM_PROVIDER_NAME}'.")
            llm_handler = None
            return False

        print(f"Initializing {LLM_PROVIDER_NAME} Provider...")
        initialized = await llm_handler.initialize(
            api_key=API_KEY_FOR_PROVIDER,
            model_config=MODEL_CONFIG,
            system_prompt=PERSONA_INSTRUCTION,
            base_url=API_BASE_URL_FOR_PROVIDER
        )

        if not initialized:
            print(f"CRITICAL: {LLM_PROVIDER_NAME} provider's initialize method returned False.")
            llm_handler = None
            return False

        print(f"{LLM_PROVIDER_NAME} Provider initialized successfully.")
        return True

    except Exception as e:
        print(f"CRITICAL: Exception caught during LLM Provider initialization setup: {e}")
        llm_handler = None
        return False


# --- キャッシュ & Deep Cache 管理 ---
# (変更なし)
async def load_cache(channel_id: int) -> List[Dict[str, Any]]:
    cache_file = os.path.join(CACHE_DIR, f"{channel_id}.json")
    if not os.path.exists(cache_file): return []
    try:
        async with aiofiles.open(cache_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content: return []
            data = json.loads(content)
            for entry in data:
                decoded_parts = []
                if 'parts' not in entry: continue
                for part in entry['parts']:
                    decoded_part = {}
                    if 'text' in part:
                        decoded_part['text'] = part['text']
                    elif 'inline_data' in part and isinstance(part.get('inline_data', {}).get('data'), str):
                        try:
                            decoded_part['inline_data'] = {
                                'mime_type': part['inline_data']['mime_type'],
                                'data': base64.b64decode(part['inline_data']['data'])
                            }
                        except Exception as e:
                            print(f"警告: キャッシュBase64デコード失敗: {e}, スキップします。 Part: {part}")
                            continue
                    else: pass
                    if decoded_part: decoded_parts.append(decoded_part)
                entry['parts'] = decoded_parts
            return data
    except json.JSONDecodeError:
        print(f"警告: キャッシュ {cache_file} が壊れています。リセットします。")
        await save_cache(channel_id, [])
        return []
    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 読み込み失敗: {e}")
        return []

async def save_cache(channel_id: int, history: List[Dict[str, Any]]):
    if not os.path.exists(CACHE_DIR):
        try: os.makedirs(CACHE_DIR)
        except Exception as e: print(f"エラー: キャッシュディレクトリ {CACHE_DIR} 作成失敗: {e}"); return

    cache_file = os.path.join(CACHE_DIR, f"{channel_id}.json")
    try:
        num_entries_to_keep = CACHE_LIMIT * 2
        history_to_save = history
        if len(history) > num_entries_to_keep:
            history_for_deep_cache = history[:-num_entries_to_keep]
            history_to_save = history[-num_entries_to_keep:]
            print(f"キャッシュ上限超過。古い履歴 ({len(history_for_deep_cache)}件) をDeep Cache更新に使用します。")
            asyncio.create_task(update_deep_cache(channel_id, history_for_deep_cache))

        encoded_history = []
        for entry in history_to_save:
            encoded_parts = []
            if 'parts' not in entry: continue
            for part in entry['parts']:
                encoded_part = {}
                if 'text' in part:
                    encoded_part['text'] = part['text']
                elif 'inline_data' in part and isinstance(part['inline_data'].get('data'), bytes):
                    try:
                        encoded_part['inline_data'] = {
                            'mime_type': part['inline_data']['mime_type'],
                            'data': base64.b64encode(part['inline_data']['data']).decode('utf-8')
                        }
                    except Exception as e:
                         print(f"警告: キャッシュBase64エンコード失敗: {e}, スキップします。 Part: {part}")
                         continue
                else: pass
                if encoded_part: encoded_parts.append(encoded_part)
            if encoded_parts: encoded_history.append({'role': entry['role'], 'parts': encoded_parts})

        async with aiofiles.open(cache_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(encoded_history, ensure_ascii=False, indent=2))
        # print(f"キャッシュ保存完了 (Channel: {channel_id}). {bot_constants.BIO_RECORD_MSG}")

    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 書き込み失敗: {e}")

async def load_deep_cache(channel_id: int) -> Optional[str]:
    deep_cache_file = os.path.join(DEEP_CACHE_DIR, f"{channel_id}.json")
    if not os.path.exists(deep_cache_file): return None
    try:
        async with aiofiles.open(deep_cache_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content: return None
            data = json.loads(content)
            return data.get("summary")
    except json.JSONDecodeError:
        print(f"警告: Deep Cache {deep_cache_file} が壊れています。リセットします。")
        await save_deep_cache(channel_id, None)
        return None
    except Exception as e:
        print(f"エラー: Deep Cache {deep_cache_file} 読み込み失敗: {e}")
        return None

async def save_deep_cache(channel_id: int, summary: Optional[str]):
    if not os.path.exists(DEEP_CACHE_DIR):
        try: os.makedirs(DEEP_CACHE_DIR)
        except Exception as e: print(f"エラー: Deep Cacheディレクトリ {DEEP_CACHE_DIR} 作成失敗: {e}"); return

    deep_cache_file = os.path.join(DEEP_CACHE_DIR, f"{channel_id}.json")
    try:
        data_to_save = {"summary": summary if summary else ""}
        async with aiofiles.open(deep_cache_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(data_to_save, ensure_ascii=False, indent=2))
        # print(f"Deep Cache 保存完了 (Channel: {channel_id}). {bot_constants.BIO_RECORD_MSG}")
    except Exception as e:
        print(f"エラー: Deep Cache {deep_cache_file} 書き込み失敗: {e}")

def _format_history_for_prompt(history: List[Dict[str, Any]]) -> str:
    formatted_lines = []
    for entry in history:
        role = entry.get('role', 'unknown').capitalize()
        text_parts = []
        for part in entry.get('parts', []):
            if 'text' in part: text_parts.append(part['text'])
            elif 'inline_data' in part: text_parts.append(f"[{part['inline_data'].get('mime_type', 'ファイル')} 添付]")
        content = " ".join(text_parts).strip()
        if content:
            max_len = 500
            if len(content) > max_len: content = content[:max_len] + "..."
            formatted_lines.append(f"{role}: {content}")
    return "\n".join(formatted_lines)

# --- Deep Cache 更新・整理関数 (LLM呼び出しを抽象化) ---
async def update_deep_cache(channel_id: int, old_history: List[Dict[str, Any]]):
    """Deep Cacheを更新する"""
    if not llm_handler: return # LLM未初期化

    print(f"Deep Cache更新開始 (Channel: {channel_id})...")
    history_text = _format_history_for_prompt(old_history)
    if not history_text.strip():
        print(f"Deep Cache更新スキップ (Channel: {channel_id}): 抽出対象テキストなし。")
        return

    extract_prompt = DEEP_CACHE_EXTRACT_PROMPT.format(history_text=history_text)
    extracted_summary = await llm_handler.generate_lowload_response(extract_prompt)

    if not extracted_summary or not extracted_summary.strip():
        print(f"Deep Cache更新: 新情報抽出失敗/空 (Channel: {channel_id})。既存キャッシュ維持。")
        return
    print(f"Deep Cache: 新情報抽出:\n{extracted_summary[:300]}...")

    existing_summary = await load_deep_cache(channel_id)
    final_summary = extracted_summary

    if existing_summary and existing_summary.strip():
        print("Deep Cache: 既存情報と新情報を統合...")
        merge_prompt = DEEP_CACHE_MERGE_PROMPT.format(
            existing_summary=existing_summary, new_summary=extracted_summary
        )
        merged_summary = await llm_handler.generate_lowload_response(merge_prompt)
        if merged_summary and merged_summary.strip():
            final_summary = merged_summary
            print(f"Deep Cache: 統合後情報生成:\n{final_summary[:300]}...")
        else:
            print("Deep Cache更新警告: 統合失敗。新情報のみ使用。")
    else:
        print("Deep Cache: 既存情報なし。抽出情報をそのまま使用。")

    await save_deep_cache(channel_id, final_summary)
    print(f"Deep Cache更新完了 (Channel: {channel_id})")

async def summarize_deep_cache(channel_id: int) -> tuple[bool, str]:
    """Deep Cacheを整理・要約する"""
    if not llm_handler:
        return False, bot_constants.ERROR_MSG_LOWLOAD_UNAVAILABLE

    print(f"Deep Cache 整理開始 (!csum, Channel: {channel_id})...")
    existing_summary = await load_deep_cache(channel_id)

    if not existing_summary or not existing_summary.strip():
        print("Deep Cache 整理スキップ: 対象データなし。")
        return False, "長期記憶(Deep Cache)には現在何も記録されていません。"

    summarize_prompt = DEEP_CACHE_SUMMARIZE_PROMPT.format(summary_to_clean=existing_summary)
    cleaned_summary = await llm_handler.generate_lowload_response(summarize_prompt)

    if not cleaned_summary or not cleaned_summary.strip():
        print("Deep Cache 整理失敗: 低負荷モデルから有効な整理結果が得られませんでした。")
        return False, bot_constants.ERROR_MSG_DEEP_CACHE_FAIL + " 整理に失敗しました。"

    await save_deep_cache(channel_id, cleaned_summary)
    print(f"Deep Cache 整理完了 (Channel: {channel_id})。")
    return True, f"長期記憶の整理が完了しました。内容は以下の通りです。\n```\n{cleaned_summary}\n```"


# --- ボタン生成・処理ヘルパー関数 (LLM呼び出しを抽象化) ---
async def generate_and_add_followup_buttons(message_to_edit: discord.Message, channel_id: int):
    """追跡質問ボタンを生成し、メッセージに追加する"""
    # 低負荷モデルがあるかどうかを llm_handler.get_model_name で確認
    if not llm_handler or not llm_handler.get_model_name('lowload'):
        print("追跡質問ボタン生成スキップ: LLMハンドラー未初期化または低負荷モデル利用不可。")
        return

    print(f"追跡質問ボタン生成試行 (Channel: {channel_id})...")
    chat_history = await load_cache(channel_id)
    if not chat_history:
        print("追跡質問ボタン生成スキップ: キャッシュ履歴なし。")
        return

    recent_history = chat_history[-2:] if len(chat_history) >= 2 else chat_history[-1:]
    recent_history_text = _format_history_for_prompt(recent_history)

    if not recent_history_text.strip():
        print("追跡質問ボタン生成スキップ: 履歴テキスト空。")
        return

    button_prompt = FOLLOW_UP_PROMPT.format(recent_history_text=recent_history_text)
    follow_up_suggestions_raw = await llm_handler.generate_lowload_response(button_prompt)

    if follow_up_suggestions_raw and "提案なし" not in follow_up_suggestions_raw:
        follow_up_prompts = [line.strip() for line in follow_up_suggestions_raw.split('\n') if line.strip()][:MAX_FOLLOW_UP_BUTTONS]
        if follow_up_prompts:
             print(f"生成された追跡質問候補: {follow_up_prompts}")
             try:
                 view = FollowUpView(original_message=message_to_edit, follow_up_prompts=follow_up_prompts)
                 await message_to_edit.edit(view=view)
                 print("追跡質問ボタンをメッセージに追加しました。")
             except discord.NotFound: print("警告: ボタン追加対象メッセージが見つかりません。")
             except discord.Forbidden: print(f"警告: {bot_constants.ERROR_MSG_PERMISSION_DENIED} (メッセージ編集)")
             except Exception as e: print(f"エラー: 追跡質問ボタンのメッセージへの追加中にエラー: {e}")
        else: print("低負荷モデルから有効な追跡質問候補が得られませんでした。")
    else: print("低負荷モデルが追跡質問の提案を生成しませんでした。")


# --- インタラクティブコンポーネント (ボタンView) ---
class FollowUpView(discord.ui.View):
    def __init__(self, original_message: discord.Message, follow_up_prompts: List[str], timeout: float = 180.0):
        super().__init__(timeout=timeout)
        self.original_message = original_message
        self.follow_up_prompts = follow_up_prompts
        for i, prompt_text in enumerate(follow_up_prompts):
            button_label = prompt_text[:80]
            button = discord.ui.Button(label=button_label, style=discord.ButtonStyle.secondary, custom_id=f"follow_up_{i}")
            button.callback = self.button_callback
            self.add_item(button)

    async def on_timeout(self):
        for item in self.children:
            if isinstance(item, discord.ui.Button): item.disabled = True
        try: await self.original_message.edit(view=None)
        except discord.NotFound: pass
        except discord.Forbidden: pass
        except Exception as e: print(f"ボタンタイムアウト後のメッセージ編集中にエラー: {e}")

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True # 誰でも押せる

    async def button_callback(self, interaction: discord.Interaction):
        """ボタンが押されたときの処理 (LLM呼び出しを抽象化)"""
        if not llm_handler: # LLM未初期化
            await interaction.response.send_message(bot_constants.ERROR_MSG_INTERNAL + " (LLM Handler)", ephemeral=True)
            return
        await interaction.response.defer()

        button_label = ""
        custom_id = interaction.data.get("custom_id", "")
        if custom_id.startswith("follow_up_"):
            try:
                index = int(custom_id.split("_")[-1])
                if 0 <= index < len(self.follow_up_prompts): button_label = self.follow_up_prompts[index]
            except (ValueError, IndexError): pass

        if not button_label:
            await interaction.followup.send(bot_constants.ERROR_MSG_BUTTON_ERROR, ephemeral=True)
            return

        print(f"追跡質問ボタン押下: '{button_label}' by {interaction.user.display_name}")

        channel_id = interaction.channel_id
        if not channel_id or not interaction.channel:
            await interaction.followup.send(bot_constants.ERROR_MSG_CHANNEL_ERROR, ephemeral=True)
            return

        # 元メッセージのボタン無効化
        for item in self.children:
            if isinstance(item, discord.ui.Button): item.disabled = True
        try: await interaction.edit_original_response(view=self)
        except discord.NotFound: pass
        except discord.Forbidden: pass
        except Exception as e: print(f"追跡質問応答前のボタン無効化エラー: {e}")

        # --- 応答生成処理 ---
        async with interaction.channel.typing():
            chat_history = await load_cache(channel_id)
            deep_cache_summary = await load_deep_cache(channel_id)
            user_entry_parts = [{'text': button_label}]

            used_model_name, response_text = await llm_handler.generate_response(
                content_parts=user_entry_parts,
                chat_history=chat_history,
                deep_cache_summary=deep_cache_summary
            )

            sent_followup_message = None
            if response_text:
                is_error_response = llm_handler._is_error_message(response_text)

                response_chunks = [response_text[i:i+1990] for i in range(0, len(response_text), 1990)]
                first_chunk = True
                for chunk in response_chunks:
                    if first_chunk:
                        sent_followup_message = await interaction.followup.send(chunk)
                        first_chunk = False
                    else:
                        await interaction.channel.send(chunk)

                if not is_error_response:
                    current_history = chat_history + [{'role': 'user', 'parts': user_entry_parts}]
                    current_history.append({'role': 'model', 'parts': [{'text': response_text}]})
                    await save_cache(channel_id, current_history)

                    if sent_followup_message:
                        await generate_and_add_followup_buttons(sent_followup_message, channel_id)
                    else:
                         print("警告: ボタン応答後メッセージ取得失敗、連続ボタン生成スキップ。")
            else:
                await interaction.followup.send(llm_handler.format_error_message(bot_constants.ERROR_TYPE_UNKNOWN, "Empty response received."))


# --- Discord BOT 設定 ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True
intents.reactions = True

client = discord.Client(intents=intents)

# --- イベントハンドラ ---
@client.event
async def on_ready():
    global discord_client_id, llm_handler
    if client.user:
        discord_client_id = str(client.user.id)
        print(f"Client ID 設定: {discord_client_id}")
    else:
        print("CRITICAL: Botユーザー情報の取得に失敗しました。")
        await client.close(); return

    if not await initialize_llm_provider():
         print("CRITICAL: LLM Provider の初期化に失敗しました。Botを終了します。")
         await client.close(); return

    if not DISCORD_TOKEN:
        print("CRITICAL: DISCORD_TOKEN が .env に未設定。")
        await client.close(); return

    for cache_dir in [CACHE_DIR, DEEP_CACHE_DIR]:
        if not os.path.exists(cache_dir):
            try: os.makedirs(cache_dir); print(f"ディレクトリ '{cache_dir}' 作成。")
            except Exception as e: print(f"警告: '{cache_dir}' 作成失敗: {e}")

    print('--------------------------------------------------')
    print("接続確認。…命令待機中。なにか御用でしょうか。")
    print(f'アカウント {client.user} としてログイン。')
    print(f'プロバイダー: {LLM_PROVIDER_NAME}')
    if llm_handler:
        primary_name = llm_handler.get_model_name('primary') or "N/A"
        secondary_name = llm_handler.get_model_name('secondary') or "N/A"
        lowload_name = llm_handler.get_model_name('lowload') or "N/A"
        print(f'モデル設定: Primary={primary_name}, Secondary={secondary_name}, Lowload={lowload_name}')
    if API_BASE_URL_FOR_PROVIDER: print(f'API Base URL: {API_BASE_URL_FOR_PROVIDER}')
    print('--------------------------------------------------')

    activity_text = f"命令待機中 ({LLM_PROVIDER_NAME}) | !poll, !timer, !csum, !cclear"
    if llm_handler and (llm_handler.get_model_name('lowload') is None or llm_handler.get_model_name('lowload') == ""):
         activity_text += " (一部機能制限)"
    await client.change_presence(activity=discord.Game(name=activity_text))

    print("Bot is ready!")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user: return
    if not message.guild: return
    if not llm_handler:
        print("Warning: LLM Handler not initialized. Skipping message processing.")
        return

    if message.content:
        content_lower = message.content.lower()

        # --- コマンド処理 ---
        if content_lower == '!csum':
            async with message.channel.typing():
                success, response_msg = await summarize_deep_cache(message.channel.id)
                await message.reply(response_msg, mention_author=False)
            return
        elif content_lower == '!cclear':
            async with message.channel.typing():
                channel_id = message.channel.id
                print(f"Deep Cache クリア実行 (!cclear, Channel: {channel_id})...")
                await save_deep_cache(channel_id, None)
                print(f"Deep Cache クリア完了 (Channel: {channel_id})。")
                await message.reply("長期記憶(Deep Cache)を初期化しました。", mention_author=False)
            return
        elif content_lower.startswith('!timer '):
            match = re.match(r'!timer\s+(\d+)\s*(分|分後|minute|minutes)\s*(.*)', message.content, re.IGNORECASE)
            if match:
                minutes = int(match.group(1))
                timer_prompt = match.group(3).strip()
                if not timer_prompt: await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 内容を指定してください。", mention_author=False); return
                if minutes <= 0: await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 時間は1分以上で指定してください。", mention_author=False); return
                await message.channel.send(f"{minutes}分後にタイマーを設定しました。 内容: 「{timer_prompt}」")
                print(f"タイマー設定: {minutes}分後, {timer_prompt}, Ch: {message.channel.name}")
                asyncio.create_task(execute_timer(message.channel, minutes, timer_prompt, message.author))
            else: await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 例: `!timer 10分 作業終了`", mention_author=False)
            return
        elif content_lower.startswith('!poll '):
            args = message.content.split(' ', 1)
            if len(args) < 2 or not args[1].strip(): await message.reply(bot_constants.ERROR_MSG_POLL_INVALID + " 内容を指定してください。", mention_author=False); return
            poll_content = args[1].strip(); parts = poll_content.split('"'); options = []
            if len(parts) >= 3 and parts[0] == '': question = parts[1].strip(); options_str = parts[2].strip(); options = [opt.strip() for opt in options_str.split() if opt.strip()]
            else: temp_parts = poll_content.split(' ', 1); question = temp_parts[0].strip();
            if len(temp_parts) > 1: options_str = temp_parts[1].strip(); options = [opt.strip() for opt in options_str.split() if opt.strip()]
            if not question or len(options) < 2 or len(options) > 10: await message.reply(bot_constants.ERROR_MSG_POLL_INVALID, mention_author=False); return
            async with message.channel.typing():
                embed = discord.Embed(title=f"投票: {question}", description="以下から選択してください。", color=discord.Color.blue())
                option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
                options_text = "".join(f"{option_emojis[i]} {option}\n" for i, option in enumerate(options))
                embed.add_field(name="選択肢", value=options_text, inline=False); embed.set_footer(text=f"作成者: {message.author.display_name}")
                try: poll_message = await message.channel.send(embed=embed); [await poll_message.add_reaction(option_emojis[i]) for i in range(len(options))]; print(f"投票作成: {question}")
                except discord.Forbidden: await message.channel.send(bot_constants.ERROR_MSG_PERMISSION_DENIED + " (メッセージ送信/リアクション追加)")
                except Exception as e: print(f"投票作成エラー: {e}"); await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " 投票作成失敗。")
            return

    # --- メンション応答 (LLM呼び出しを抽象化) ---
    if client.user and client.user.mentioned_in(message):
        async with message.channel.typing():
            channel_id = message.channel.id

            # 1. プロンプトと添付ファイルの準備
            mention_strings = [f'<@!{client.user.id}>', f'<@{client.user.id}>']
            text_content = message.content if message.content else ""
            for mention in mention_strings: text_content = text_content.replace(mention, '')
            text_content = text_content.strip()
            use_channel_history = '!his' in text_content
            if use_channel_history: text_content = text_content.replace('!his', '').strip(); print("履歴参照フラグ (!his) 検出。キャッシュ無視。")

            request_parts: List[Dict[str, Any]] = []
            if text_content: request_parts.append({'text': text_content})
            attached_files_data_for_cache = []
            file_error_occurred_once = False
            if message.attachments:
                print(f"{len(message.attachments)}個の添付ファイルを検出。")
                max_images = 5; image_count = 0
                for attachment in message.attachments:
                    if attachment.size > 50 * 1024 * 1024:
                        if not file_error_occurred_once: await message.channel.send(bot_constants.ERROR_MSG_FILE_SIZE_LIMIT); file_error_occurred_once = True; print(f"警告: 添付 '{attachment.filename}' サイズ超過。")
                        continue
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        image_count += 1
                        if image_count > max_images:
                            if not file_error_occurred_once: await message.channel.send(bot_constants.ERROR_MSG_MAX_IMAGE_SIZE); file_error_occurred_once = True; print(f"警告: 画像数超過。")
                            continue
                    try:
                        file_bytes = await attachment.read()
                        mime_type = attachment.content_type
                        if mime_type is None: mime_type, _ = mimetypes.guess_type(attachment.filename); mime_type = mime_type or 'application/octet-stream'
                        if mime_type and mime_type.startswith('text/plain'): mime_type = 'text/plain'

                        supported_prefixes = ('image/', 'text/')
                        if not any(mime_type.startswith(prefix) for prefix in supported_prefixes):
                             print(f"警告: 未対応MIME '{mime_type}' スキップ。")
                             if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_ATTACHMENT_UNSUPPORTED} ({mime_type})"); file_error_occurred_once = True
                             continue

                        request_parts.append({'inline_data': {'mime_type': mime_type, 'data': file_bytes}})
                        attached_files_data_for_cache.append({'mime_type': mime_type, 'data': file_bytes})
                        print(f"添付 '{attachment.filename}' ({mime_type}) 追加。")
                    except discord.HTTPException as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_IMAGE_READ_FAIL} (Discord Error)"); file_error_occurred_once = True; print(f"エラー: 添付 '{attachment.filename}' 読込失敗 (Discord): {e}")
                    except Exception as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_INTERNAL} (ファイル処理エラー)"); file_error_occurred_once = True; print(f"エラー: 添付 '{attachment.filename}' 処理中エラー: {e}")

            if not request_parts: print("応答可能なテキスト/添付なし、スキップ。"); return

            # 2. 履歴の準備
            chat_history: List[Dict[str, Any]] = []
            if use_channel_history:
                print(f"チャンネル履歴 ({HISTORY_LIMIT}件) 取得中...")
                try:
                    history_messages = [msg async for msg in message.channel.history(limit=HISTORY_LIMIT + 1)]
                    history_messages.reverse(); history_messages = history_messages[:-1]
                    for msg in history_messages:
                        role = 'model' if msg.author == client.user else 'user'
                        msg_parts = []; txt = msg.content or ""
                        if msg.attachments: txt += " " + " ".join([f"[{att.filename} 添付]" for att in msg.attachments])
                        if txt: msg_parts.append({'text': txt.strip()})
                        if msg_parts: chat_history.append({'role': role, 'parts': msg_parts})
                    print(f"チャンネル履歴から {len(chat_history)} 件整形。")
                except discord.Forbidden: await message.channel.send(bot_constants.ERROR_MSG_PERMISSION_DENIED + " (履歴読取)"); return
                except Exception as e: await message.channel.send(bot_constants.ERROR_MSG_HISTORY_READ_FAIL); print(f"エラー: 履歴取得失敗: {e}"); return
            else:
                print(f"チャンネル {channel_id} キャッシュ読込中...")
                chat_history = await load_cache(channel_id)
                print(f"キャッシュから {len(chat_history)} 件読込。")

            # 3. Deep Cacheの準備
            deep_cache_summary = await load_deep_cache(channel_id)
            if deep_cache_summary: print("Deep Cache情報読込。")
            else: print("Deep Cache情報なし。")

            # 4. LLM API呼び出し (抽象化)
            # generate_response は内部で適切なモデル名を使って変換を行う
            used_model_name, response_text = await llm_handler.generate_response(
                content_parts=request_parts,
                chat_history=chat_history,
                deep_cache_summary=deep_cache_summary
            )

            # 5. 応答送信
            sent_message = None
            if response_text:
                is_error_response_text = llm_handler._is_error_message(response_text) # エラー判定用

                if not is_error_response_text: # エラーメッセージでない場合のみ分割送信
                    response_chunks = [response_text[i:i+1990] for i in range(0, len(response_text), 1990)]
                    first_chunk = True
                    for chunk in response_chunks:
                        if first_chunk: sent_message = await message.reply(chunk, mention_author=False); first_chunk = False
                        else: await message.channel.send(chunk)
                else:
                    # エラーメッセージの場合はそのまま送信
                    await message.reply(response_text, mention_author=False)


            # 6. キャッシュ更新
            # エラー判定は LLM Provider が返すメッセージに基づいて行う
            is_error_response = response_text is None or llm_handler._is_error_message(response_text)

            if not is_error_response and not use_channel_history:
                user_entry_parts = []
                if text_content: user_entry_parts.append({'text': text_content})
                user_entry_parts.extend({'inline_data': file_info} for file_info in attached_files_data_for_cache)
                if user_entry_parts:
                     current_history = chat_history + [{'role': 'user', 'parts': user_entry_parts}]
                     # モデル応答はテキストのみキャッシュ (画像応答はキャッシュしない)
                     if response_text and not llm_handler._is_error_message(response_text):
                         current_history.append({'role': 'model', 'parts': [{'text': response_text}]})
                     await save_cache(channel_id, current_history)


            # 7. 追跡質問ボタン生成
            if sent_message and not is_error_response:
                 await generate_and_add_followup_buttons(sent_message, channel_id)


# --- タイマー実行関数 (LLM呼び出しを抽象化) ---
async def execute_timer(channel: discord.TextChannel, minutes: int, prompt: str, author: discord.User):
    if not llm_handler: return # LLM未初期化
    await asyncio.sleep(minutes * 60)
    print(f"タイマー実行: {minutes}分経過, {prompt}, Ch: {channel.name}")
    async with channel.typing():
        mention = author.mention
        base_message = f"{mention} 指定時刻です。タイマーの内容: 「{prompt}」"

        timer_execution_prompt = f"「{prompt}」というリマインダーの指定時刻になりました。ユーザー ({author.display_name}) に向けて、簡潔な補足メッセージを生成してください。"
        _used_model, response_text = await llm_handler.generate_response([{'text': timer_execution_prompt}], chat_history=None, deep_cache_summary=None)

        full_message = base_message
        if response_text and not llm_handler._is_error_message(response_text):
            full_message += f"\n\n{response_text}"
        elif response_text:
             full_message += f"\n\n補足情報の生成に失敗しました。({response_text})"

        for i in range(0, len(full_message), 1990): await channel.send(full_message[i:i+1990])


# --- BOT起動 ---
if __name__ == "__main__":
    try:
        import aiofiles
        import dotenv
        if LLM_PROVIDER_NAME == 'GEMINI':
            import google.generativeai
        elif LLM_PROVIDER_NAME == 'MISTRAL':
             import openai
        else:
             print(f"警告: 未対応のLLM_PROVIDER '{LLM_PROVIDER_NAME}' が設定されています。")
             pass

    except ImportError as e:
        print(f"CRITICAL: 必要なライブラリが不足しています: {e}")
        print("実行前に `pip install aiofiles python-dotenv discord.py google-generativeai openai` を実行してください。")
        exit(1)
    except Exception as e:
         print(f"CRITICAL: 依存ライブラリチェック中に予期せぬエラー: {e}")
         exit(1)

    try:
        print("BOT起動処理開始...")
        client.run(DISCORD_TOKEN)
    except discord.LoginFailure: print("CRITICAL: 不正なDiscordトークン。")
    except discord.PrivilegedIntentsRequired: print("CRITICAL: 特権インテント(Message Content等)無効。Discord Developer Portal確認要。")
    except Exception as e:
        print(f"CRITICAL: BOT実行中に予期せぬエラーが発生しました: {e}")
        import traceback; traceback.print_exc()