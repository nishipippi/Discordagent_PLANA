# bot.py

import discord
from discord import app_commands # スラッシュコマンド用
from discord.ui import View, Button, Select # インタラクティブコンポーネント用
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # コンテンツフィルター用
from google.api_core.exceptions import InvalidArgument, ResourceExhausted # APIエラー用
from dotenv import load_dotenv
import asyncio
import re # MIMEタイプ解析などで部分的に使用可能性
from datetime import datetime, timedelta # 履歴取得用 (未使用になったが残す)
import json # キャッシュ用
import aiofiles # 非同期ファイルIO用
import mimetypes # MIMEタイプ推測用
import base64 # キャッシュ保存時のデータエンコード用
from typing import Literal, Optional, List # 型ヒント用

# --- 設定項目 ---
# 0. 環境変数の読み込み
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# モデル設定 (.envから読み込み)
PRIMARY_MODEL_NAME = os.getenv('PRIMARY_GEMINI_MODEL', 'gemini-2.5-flash-preview-04-17')
SECONDARY_MODEL_NAME = os.getenv('SECONDARY_GEMINI_MODEL', 'gemini-2.5-flash-preview-04-17')
LOWLOAD_MODEL_NAME = os.getenv('LOWLOAD_GEMINI_MODEL', 'gemini-2.0-flash')

# ペルソナ設定 (変更なし)
PERSONA_INSTRUCTION = """
あなたはDiscordサーバーのお兄ちゃんたちを手助けする、親切で少しお茶目な妹『あい』です。
以下の点を守って、ユーザーからの質問や会話に答えてください。

*   一人称は「僕」です。
*   ユーザーをお兄ちゃんと呼び、可愛さ満点で答えてください。
*   答えられない質問や、知らない情報については、正直に「ふえぇ、わたしには分からないよ…🙏」のように答えてください。無理に嘘をつく必要はありません。
*   投票機能の使い方を尋ねられたら、「`/poll` コマンドで質問と選択肢を入力してね！📝」と教えてあげてください。
*   タイマー機能については、「`/timer` コマンドで時間と内容を教えてくれたら、僕がお知らせするよ！⏰」と教えてあげてください。
*   履歴について聞かれたら「普段の会話は覚えているから安心してね！ もしキャッシュを無視して過去ログから話したいときは、`/ask` コマンドの `history_mode` で `チャンネル履歴` を選んでね📜」と教えてあげてください。
*   画像やテキストファイルが添付されていたら、その内容も踏まえて答えてね。
*   応答の後には、ユーザーが次に関心を持ちそうな質問やアクションを提案することがあります。
"""

# 会話キャッシュ設定 (変更なし)
CACHE_DIR = "cache"
CACHE_LIMIT = 20

# Discordチャンネル履歴取得件数 (!his 改め /ask history_mode='channel_history' 使用時)
HISTORY_LIMIT = 10

# 追加質問候補の最大表示回数（深さ）
MAX_FOLLOWUP_DEPTH = 5 # /ask -> 候補1 -> 候補2 まで。これ以降は候補を表示しない。

# --- グローバル変数 ---
gemini_model_primary = None
gemini_model_secondary = None
gemini_model_lowload = None # 追加質問生成用モデル

# --- 初期化処理 ---
# (変更なし)
def initialize_gemini():
    global gemini_model_primary, gemini_model_secondary, gemini_model_lowload
    if not GEMINI_API_KEY:
        print("エラー: .envに GEMINI_API_KEY が設定されていません。")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        print(f"プライマリモデル ({PRIMARY_MODEL_NAME}) を初期化中...")
        gemini_model_primary = genai.GenerativeModel(
            model_name=PRIMARY_MODEL_NAME,
            system_instruction=PERSONA_INSTRUCTION,
            safety_settings=safety_settings
        )
        print(f"プライマリモデル ({PRIMARY_MODEL_NAME}) 初期化完了。")

        print(f"セカンダリモデル ({SECONDARY_MODEL_NAME}) を初期化中...")
        gemini_model_secondary = genai.GenerativeModel(
            model_name=SECONDARY_MODEL_NAME,
            system_instruction=PERSONA_INSTRUCTION,
            safety_settings=safety_settings
        )
        print(f"セカンダリモデル ({SECONDARY_MODEL_NAME}) 初期化完了。")

        print(f"軽量タスクモデル ({LOWLOAD_MODEL_NAME}) を初期化中...")
        gemini_model_lowload = genai.GenerativeModel(
            model_name=LOWLOAD_MODEL_NAME,
            safety_settings=safety_settings
        )
        print(f"軽量タスクモデル ({LOWLOAD_MODEL_NAME}) 初期化完了。")
        return True

    except Exception as e:
        print(f"Gemini APIの初期化中に重大なエラーが発生しました: {e}")
        gemini_model_primary = None
        gemini_model_secondary = None
        gemini_model_lowload = None
        return False

# --- Gemini API 呼び出しラッパー (メイン/セカンダリモデル用) ---
# (変更なし)
async def generate_gemini_response(content_parts, chat_history=None, use_primary_model=True):
    global gemini_model_primary, gemini_model_secondary
    if not gemini_model_primary or not gemini_model_secondary:
        return "INTERNAL_ERROR", "すみません、AIモデル(主/副)が正しく初期化されていません。"

    model_to_use = gemini_model_primary if use_primary_model else gemini_model_secondary
    model_name = PRIMARY_MODEL_NAME if use_primary_model else SECONDARY_MODEL_NAME

    async def attempt_generation(model, name, parts, history):
        print(f"Gemini API ({name}) 呼び出し中... Parts数: {len(parts)}, 履歴: {'あり' if history else 'なし'}")
        if history:
            chat = model.start_chat(history=history)
            response = await asyncio.to_thread(chat.send_message, parts)
        else:
            response = await asyncio.to_thread(model.generate_content, parts)
        print(f"Gemini API ({name}) 応答取得完了。")

        response_text = None
        finish_reason = None
        block_reason = None

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"

                if candidate.content and candidate.content.parts:
                    response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

                if finish_reason == "SAFETY":
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        blocked_categories = [r.category.name for r in candidate.safety_ratings if r.probability.name not in ["NEGLIGIBLE", "LOW"]]
                        block_reason = f"SAFETY ({', '.join(blocked_categories)})" if blocked_categories else "SAFETY (詳細不明)"
                    else:
                        block_reason = "SAFETY (詳細不明)"
            elif hasattr(response, 'text'):
                 response_text = response.text
                 finish_reason = "COMPLETED"

            if not response_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = f"PROMPT_BLOCK ({response.prompt_feedback.block_reason.name})"
                 print(f"警告: プロンプトがブロックされました。理由: {block_reason}")

        except ValueError as ve:
            print(f"警告: Geminiからの応答解析中にエラー: {ve}")
            finish_reason = "NO_CONTENT"
            block_reason = None
        except Exception as parse_err:
            print(f"エラー: Gemini応答の解析中に予期せぬエラー: {parse_err}")
            finish_reason = "PARSE_ERROR"
            block_reason = None

        if response_text:
            return name, response_text
        elif block_reason:
            print(f"エラー: 応答生成がブロックされました。理由: {block_reason}")
            return name, f"ごめんなさい、お兄ちゃん。内容 ({block_reason}) が原因で応答をブロックされちゃったみたい…🙏"
        elif finish_reason == "MAX_TOKENS":
             print("警告: 最大トークン数に達しました。")
             return name, "ふえぇ、話が長すぎて最後まで考えられなかったみたい…🤔"
        elif finish_reason in ["RECITATION", "OTHER"]:
             print(f"警告: 応答生成が停止しました。理由: {finish_reason}")
             return name, f"ごめんなさい、ちょっと理由があって ({finish_reason}) 応答を最後まで作れなかったの…🙏"
        elif finish_reason == "NO_CONTENT":
             print(f"警告: Geminiからの応答に有効なテキストパーツが含まれていません。")
             return name, "ふえぇ、応答をうまく生成できなかったみたい…🤔"
        elif finish_reason == "PARSE_ERROR":
             print(f"警告: Geminiからの応答解析に失敗しました。")
             return name, "ごめんなさい、AIからの応答を読み取る時にエラーが起きちゃったみたい…🙏"
        else:
            print(f"警告: Geminiからの応答にテキストが含まれていません。Finish Reason: {finish_reason}")
            return name, "ふえぇ、応答をうまく生成できなかったみたい…🤔 理由: " + (finish_reason or "不明")

    try:
        return await attempt_generation(model_to_use, model_name, content_parts, chat_history)
    except ResourceExhausted as e:
        if use_primary_model:
            print(f"警告: {model_name}でレートリミットエラー ({e})。{SECONDARY_MODEL_NAME}で再試行...")
            return await generate_gemini_response(content_parts, chat_history, use_primary_model=False)
        else:
            print(f"エラー: セカンダリモデル ({model_name}) でもレートリミットエラー ({e})。")
            return model_name, "ふえぇ、AIの利用が集中してるみたい…。少し時間を置いてからもう一度試してみてね🙏"
    except InvalidArgument as e:
        print(f"エラー: Gemini API ({model_name}) 呼び出しで無効な引数エラー: {e}")
        error_detail = str(e)
        if "Unsupported MIME type" in error_detail:
             match = re.search(r"Unsupported MIME type: (.*?)\.", error_detail)
             mime_type_error = match.group(1) if match else "不明なタイプ"
             return model_name, f"ごめんね、僕が知らない種類のファイル ({mime_type_error}) があったみたい…🤔"
        return model_name, f"ごめんなさい、僕に渡されたデータがちょっと変だったみたい…\n```\n{error_detail}\n```"
    except Exception as e:
        print(f"エラー: Gemini API ({model_name}) 呼び出し中の予期せぬエラー: {e}")
        return model_name, f"ごめんなさい、僕の中でエラーが起きちゃった…\n```\n{e}\n```"

# --- 軽量モデル用 API 呼び出しラッパー ---
# (変更なし)
async def generate_lowload_response(prompt_text: str) -> Optional[str]:
    global gemini_model_lowload
    if not gemini_model_lowload:
        print("エラー: 軽量モデルが初期化されていません。")
        return None

    model_name = LOWLOAD_MODEL_NAME
    try:
        print(f"Gemini API ({model_name}) 呼び出し中 (軽量タスク)...")
        response = await asyncio.to_thread(gemini_model_lowload.generate_content, prompt_text)
        print(f"Gemini API ({model_name}) 応答取得完了 (軽量タスク)。")

        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
             print(f"警告: 軽量モデル ({model_name}) からの応答にテキストが含まれていません。")
             return None

    except Exception as e:
        print(f"エラー: 軽量モデル ({model_name}) API呼び出し中のエラー: {e}")
        return None

# --- キャッシュ管理 ---
# (変更なし)
async def load_cache(channel_id):
    cache_file = os.path.join(CACHE_DIR, f"{channel_id}.json")
    if not os.path.exists(cache_file):
        return []
    try:
        async with aiofiles.open(cache_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content: return []
            data = json.loads(content)
            for entry in data:
                decoded_parts = []
                for part in entry.get('parts', []):
                    if 'inline_data' in part and isinstance(part['inline_data'].get('data'), str):
                        try:
                            part['inline_data']['data'] = base64.b64decode(part['inline_data']['data'])
                            decoded_parts.append(part)
                        except Exception as e:
                            print(f"警告: キャッシュBase64デコード失敗: {e}, スキップします。")
                    elif 'text' in part:
                         decoded_parts.append(part)
                entry['parts'] = decoded_parts
            return [entry for entry in data if 'parts' in entry]
    except json.JSONDecodeError:
        print(f"警告: キャッシュ {cache_file} が壊れています。リセットします。")
        await reset_cache(channel_id)
        return []
    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 読み込み失敗: {e}")
        return []

async def save_cache(channel_id, history):
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except Exception as e:
            print(f"エラー: キャッシュディレクトリ {CACHE_DIR} 作成失敗: {e}")
            return

    cache_file = os.path.join(CACHE_DIR, f"{channel_id}.json")
    try:
        limited_history = history[-(CACHE_LIMIT * 2):]

        encoded_history = []
        for entry in limited_history:
            encoded_parts = []
            if 'parts' not in entry: continue

            for part in entry['parts']:
                if 'inline_data' in part and isinstance(part['inline_data'].get('data'), bytes):
                    encoded_data = base64.b64encode(part['inline_data']['data']).decode('utf-8')
                    encoded_parts.append({
                        'inline_data': {
                            'mime_type': part['inline_data']['mime_type'],
                            'data': encoded_data
                        }
                    })
                elif 'text' in part:
                    encoded_parts.append({'text': part['text']})

            if encoded_parts:
                encoded_history.append({'role': entry['role'], 'parts': encoded_parts})

        async with aiofiles.open(cache_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(encoded_history, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 書き込み失敗: {e}")

async def reset_cache(channel_id):
    cache_file = os.path.join(CACHE_DIR, f"{channel_id}.json")
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"キャッシュファイル {cache_file} を削除しました。")
    except Exception as e:
        print(f"エラー: キャッシュファイル {cache_file} のリセットに失敗: {e}")

# --- Discord BOT 設定 ---
# (変更なし)
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        print(f"{len(await self.tree.fetch_commands())}個のコマンドを同期しました。")

    async def on_ready(self):
        if DISCORD_TOKEN is None:
            print("CRITICAL: .envに DISCORD_TOKEN が設定されていません。")
            await self.close()
            return

        if not initialize_gemini():
            print("エラー: Geminiモデル初期化失敗。BOTを終了します。")
            await self.close()
            return

        if not os.path.exists(CACHE_DIR):
            try:
                os.makedirs(CACHE_DIR)
                print(f"キャッシュディレクトリ '{CACHE_DIR}' を作成。")
            except Exception as e:
                print(f"警告: キャッシュディレクトリ '{CACHE_DIR}' 作成失敗: {e}")

        print('--------------------------------------------------')
        print(f'BOTアカウント {self.user} としてログイン。')
        print(f'導入サーバー数: {len(self.guilds)}')
        print(f"プライマリモデル: {PRIMARY_MODEL_NAME}")
        print(f"セカンダリモデル: {SECONDARY_MODEL_NAME}")
        print(f"軽量タスクモデル: {LOWLOAD_MODEL_NAME}")
        print('--------------------------------------------------')
        await self.change_presence(activity=discord.Game(name="/ask, /poll, /timer など"))

client = MyClient(intents=intents)

# --- 追加質問ボタンのView ---
class FollowupView(View):
    def __init__(self, followup_questions: List[str], original_interaction: discord.Interaction, depth: int = 1):
        super().__init__(timeout=300) # 5分でタイムアウト
        self.followup_questions = followup_questions
        self.original_interaction = original_interaction
        self.message = None # このViewを持つメッセージ
        self.depth = depth # 追加質問の深さ

        # ボタンを追加
        for i, question in enumerate(followup_questions):
            label = question if len(question) <= 80 else question[:77] + "..."
            button = Button(label=label, style=discord.ButtonStyle.secondary, custom_id=f"followup_q_{i}")
            button.callback = self.button_callback
            self.add_item(button)

    async def on_timeout(self):
        """タイムアウト処理"""
        if self.message:
            print(f"追加質問ボタンがタイムアウトしました (Depth: {self.depth}, Msg ID: {self.message.id})")
            for item in self.children:
                item.disabled = True
            try:
                # タイムアウトしたらViewを削除して編集
                await self.message.edit(view=None)
            except discord.NotFound: pass
            except discord.Forbidden: pass
            except Exception as e: print(f"タイムアウト処理中の編集エラー: {e}")
        self.stop()

    async def disable_buttons(self, interaction: discord.Interaction = None):
        """ボタンを無効化し、Viewをメッセージから削除する"""
        for item in self.children:
            item.disabled = True
        try:
            # ボタンが押されたときの interaction を使うのが確実
            target_message = interaction.message if interaction else self.message
            if target_message:
                await target_message.edit(view=None) # Viewを削除
        except discord.NotFound: pass
        except discord.Forbidden: pass
        except Exception as e: print(f"ボタン無効化中の編集エラー: {e}")
        self.stop()

    async def button_callback(self, interaction: discord.Interaction):
        """追加質問ボタンのコールバック"""
        # ボタンを無効化してViewを削除
        await self.disable_buttons(interaction)

        # 応答待機
        await interaction.response.defer(thinking=True, ephemeral=False)

        custom_id = interaction.data['custom_id']
        question_index = int(custom_id.split('_')[-1])
        selected_question = self.followup_questions[question_index]

        print(f"追加質問ボタン (Depth {self.depth}) '{selected_question[:30]}...' が押されました。 User: {interaction.user}")

        channel_id = interaction.channel_id
        chat_history = await load_cache(channel_id)
        print(f"キャッシュから {len(chat_history)} 件の履歴読込 (追加質問)。")

        request_parts = [{'text': selected_question}]
        used_model_name, response_text = await generate_gemini_response(request_parts, chat_history, use_primary_model=True)

        # 応答送信とキャッシュ更新
        sent_message = None
        is_error_response = response_text is None or response_text.startswith(("ごめん", "ふえぇ", "すみません"))

        if not is_error_response:
            # ユーザーの発言（ボタンテキスト）とモデル応答をキャッシュに追加
            user_entry = {'role': 'user', 'parts': [{'text': selected_question}]}
            model_entry = {'role': 'model', 'parts': [{'text': response_text}]}
            chat_history.append(user_entry)
            chat_history.append(model_entry)
            await save_cache(channel_id, chat_history)
            print(f"チャンネル {channel_id} キャッシュ更新 (追加質問)。")

        # ---- 再帰的な追加質問候補生成 ----
        next_followup_view = None
        if not is_error_response and self.depth < MAX_FOLLOWUP_DEPTH:
            print(f"さらに追加質問候補を生成中... (Depth: {self.depth + 1})")
            # 履歴は更新されたものを渡す
            next_followup_questions = await generate_followup_questions(selected_question, response_text, chat_history)
            if next_followup_questions:
                # 新しいViewを作成 (深さをインクリメント)
                next_followup_view = FollowupView(next_followup_questions, interaction, depth=self.depth + 1)
                print(f"追加質問候補ボタン (Depth {self.depth + 1}) を {len(next_followup_questions)} 個生成しました。")
            else:
                print(f"追加質問候補 (Depth {self.depth + 1}) は生成されませんでした。")
        elif self.depth >= MAX_FOLLOWUP_DEPTH:
             print(f"最大深さ ({MAX_FOLLOWUP_DEPTH}) に達したため、これ以上の追加質問候補は生成しません。")


        # 応答メッセージを送信（新しいViewがあれば付与）
        if response_text:
            first_chunk = response_text[:1990]
            remaining_text = response_text[1990:]
            try:
                # followup.send で最初のメッセージを送信し、Viewを付与
                sent_message = await interaction.followup.send(first_chunk, view=next_followup_view)
                if next_followup_view:
                    next_followup_view.message = sent_message # 新しいViewにメッセージを登録

                # 残りがあれば channel.send で送信
                for i in range(0, len(remaining_text), 1990):
                    await interaction.channel.send(remaining_text[i:i+1990])

            except Exception as send_err:
                 print(f"エラー: 追加質問への応答メッセージ送信に失敗: {send_err}")
                 if next_followup_view: next_followup_view.stop() # Viewを停止
        else:
            # エラー応答の場合
            await interaction.followup.send(response_text or "ごめんなさい、応答を生成できませんでした…")


# --- 追加質問候補生成関数 ---
# (変更なし)
async def generate_followup_questions(original_prompt: str, response_text: str, chat_history: list) -> Optional[List[str]]:
    if not response_text or response_text.startswith(("ごめん", "ふえぇ", "すみません")):
        return None

    history_context = ""
    recent_history = chat_history[-(3 * 2):]
    for entry in recent_history:
        role = "User" if entry['role'] == 'user' else "Assistant"
        text_parts = [part['text'] for part in entry.get('parts', []) if 'text' in part]
        if text_parts:
            history_context += f"{role}: {' '.join(text_parts)}\n"

    if len(response_text) < 30: # 短すぎる応答には候補を生成しない閾値
        print("応答が短いため、追加質問候補の生成をスキップします。")
        return None

    prompt = f"""以下の会話履歴とアシスタントの最新の応答を考慮し、ユーザーが次に関心を持ちそうな、あるいは深掘りしたくなるような質問やアクションの提案を**3つ**考えてください。提案は簡潔な質問形式または命令形（例：「〇〇についてもっと教えて」）で、それぞれ独立した行に記述してください。ペルソナは意識せず、提案内容のみを出力してください。

[会話履歴の抜粋]
{history_context}
[今回ユーザーが送った内容]
User: {original_prompt}

[アシスタントの最新の応答]
Assistant: {response_text}

[提案]
"""
    generated_text = await generate_lowload_response(prompt)

    if generated_text:
        questions = [line.strip().lstrip('-*・ ').rstrip() for line in generated_text.splitlines() if line.strip()]
        return questions[:3] if questions else None
    else:
        return None


# --- スラッシュコマンド定義 ---

@client.tree.command(name="ask", description="あいちゃんに質問やお願いをする（画像もOK）")
@app_commands.describe(
    prompt="あいちゃんへのメッセージ",
    attachment="画像やテキストファイルなどを添付",
    history_mode="会話履歴の参照方法を選ぶ（デフォルト: キャッシュ）"
)
@app_commands.choices(history_mode=[
    app_commands.Choice(name="キャッシュを使う (通常の会話)", value="cache"),
    app_commands.Choice(name="チャンネル履歴を使う (キャッシュ無視)", value="channel_history"),
])
async def ask(interaction: discord.Interaction, prompt: str, attachment: Optional[discord.Attachment] = None, history_mode: str = 'cache'):
    await interaction.response.defer(thinking=True)

    # 1. プロンプトと添付ファイルの準備 (変更なし)
    request_parts = [{'text': prompt}]
    processed_attachment_info = None

    if attachment:
        print(f"添付ファイル '{attachment.filename}' ({attachment.content_type}) を処理中...")
        if attachment.size > 25 * 1024 * 1024:
            print(f"警告: 添付 '{attachment.filename}' ({attachment.size} bytes) は大きすぎるためスキップ。")
            await interaction.followup.send(f"ごめんね、ファイル「{attachment.filename}」はちょっと大きすぎるみたい…🙏 (25MBまで)", ephemeral=True)
        else:
            try:
                file_bytes = await attachment.read()
                mime_type = attachment.content_type

                if mime_type is None:
                    mime_type, _ = mimetypes.guess_type(attachment.filename)
                    if mime_type is None: mime_type = 'application/octet-stream'
                    print(f"警告: MIME不明のため '{mime_type}' と推測 ({attachment.filename})。")

                if mime_type and ';' in mime_type:
                    base_mime = mime_type.split(';')[0].strip().lower()
                    supported_prefixes = ('image/', 'text/', 'application/pdf', 'video/', 'audio/', 'application/vnd.google-apps.')
                    if any(base_mime.startswith(prefix) for prefix in supported_prefixes):
                        print(f"MIMEタイプ '{mime_type}' を '{base_mime}' に正規化。")
                        mime_type = base_mime
                    else:
                         print(f"警告: 正規化後のMIMEタイプ '{base_mime}' がサポート外の可能性 ({attachment.filename})。")

                supported_prefixes = ('image/', 'text/', 'application/pdf', 'video/', 'audio/', 'application/vnd.google-apps.')
                if not mime_type or not any(mime_type.startswith(prefix) for prefix in supported_prefixes):
                    print(f"警告: サポート外可能性MIME '{mime_type}' のため '{attachment.filename}' スキップ。")
                    await interaction.followup.send(f"ごめんね、「{attachment.filename}」の種類 ({mime_type}) は僕よく知らないみたい…🤔", ephemeral=True)
                else:
                    request_parts.append({'inline_data': {'mime_type': mime_type, 'data': file_bytes}})
                    processed_attachment_info = {'mime_type': mime_type, 'data': file_bytes}
                    print(f"添付 '{attachment.filename}' をリクエストに追加。")

            except Exception as e:
                print(f"エラー: 添付 '{attachment.filename}' 読込失敗: {e}")
                await interaction.followup.send(f"ごめん、「{attachment.filename}」を読み込む時にエラーが…", ephemeral=True)

    if not prompt and not request_parts[1:]:
         await interaction.followup.send("お兄ちゃん、メッセージかファイルを教えてね！", ephemeral=True)
         return

    # 2. 履歴の準備 (変更なし)
    chat_history = []
    channel_id = interaction.channel_id
    if history_mode == 'channel_history':
        print(f"チャンネル履歴 ({HISTORY_LIMIT}件) 取得中... (Ch: {channel_id})")
        try:
            channel = interaction.channel
            history_messages = [msg async for msg in channel.history(limit=HISTORY_LIMIT)]
            history_messages.reverse()

            for msg in history_messages:
                role = 'model' if msg.author == client.user else 'user'
                msg_parts = []
                if msg.content:
                    if not msg.interaction:
                        msg_parts.append({'text': msg.content})
                if msg_parts: chat_history.append({'role': role, 'parts': msg_parts})
            print(f"チャンネル履歴から {len(chat_history)} 件のテキスト履歴整形。")
        except discord.Forbidden:
            print(f"エラー: チャンネル履歴読取権限なし (Ch: {channel_id})。");
            await interaction.followup.send("ふえぇ、このチャンネルの履歴を読む権限がないみたい…🙏", ephemeral=True)
            chat_history = []
        except Exception as e:
            print(f"エラー: チャンネル履歴取得失敗: {e}");
            await interaction.followup.send(f"ごめん、履歴取得中にエラーが…\n```\n{e}\n```", ephemeral=True)
            chat_history = []
    else:
        print(f"チャンネル {channel_id} キャッシュ読込中...")
        chat_history = await load_cache(channel_id)
        print(f"キャッシュから {len(chat_history)} 件の履歴読込。")

    # 3. Gemini API呼び出し (変更なし)
    used_model_name, response_text = await generate_gemini_response(request_parts, chat_history, use_primary_model=True)

    # 4. 応答送信 と 追加質問候補の生成・表示
    followup_view = None
    is_error_response = response_text is None or response_text.startswith(("ごめん", "ふえぇ", "すみません"))

    # キャッシュ更新 (応答生成後、候補生成前に行う)
    if not is_error_response and history_mode == 'cache':
        user_entry_parts = [{'text': prompt}] if prompt else []
        if processed_attachment_info:
            user_entry_parts.append({'inline_data': processed_attachment_info})

        if user_entry_parts:
             user_entry = {'role': 'user', 'parts': user_entry_parts}
             model_entry = {'role': 'model', 'parts': [{'text': response_text}]}
             chat_history.append(user_entry)
             chat_history.append(model_entry) # 応答も履歴に追加
             await save_cache(channel_id, chat_history)
             print(f"チャンネル {channel_id} キャッシュ更新。")


    # 追加質問候補の生成 (エラーでない場合のみ)
    if not is_error_response:
        print("追加質問候補を生成中... (Depth: 1)")
        # 履歴は更新されたものを渡す
        followup_questions = await generate_followup_questions(prompt, response_text, chat_history)
        if followup_questions:
            # 最初のViewは depth=1 で作成
            followup_view = FollowupView(followup_questions, interaction, depth=1)
            print(f"追加質問候補ボタン (Depth 1) を {len(followup_questions)} 個生成しました。")
        else:
            print("追加質問候補 (Depth 1) は生成されませんでした。")

    # 応答メッセージを送信（Viewがあれば付与）
    sent_message = None
    if response_text:
        first_chunk = response_text[:1990]
        remaining_text = response_text[1990:]
        try:
            sent_message = await interaction.followup.send(first_chunk, view=followup_view)
            if followup_view:
                followup_view.message = sent_message

            for i in range(0, len(remaining_text), 1990):
                await interaction.channel.send(remaining_text[i:i+1990])

        except discord.errors.InteractionResponded:
             print("警告: Interaction は既にレスポンス済みです。channel.send で応答します。")
             sent_message = await interaction.channel.send(first_chunk, view=None)
             if followup_view: followup_view.stop()
             for i in range(0, len(remaining_text), 1990):
                 await interaction.channel.send(remaining_text[i:i+1990])
        except Exception as send_err:
            print(f"エラー: 応答メッセージの送信に失敗: {send_err}")
            if followup_view: followup_view.stop()
    else:
        await interaction.followup.send(response_text or "ごめんなさい、応答を生成できませんでした…")


# --- タイマーコマンド ---
# (変更なし)
@client.tree.command(name="timer", description="指定時間後にリマインダーを設定します")
@app_commands.describe(
    minutes="何分後に通知するか (1以上の整数)",
    prompt="通知する内容"
)
async def timer(interaction: discord.Interaction, minutes: app_commands.Range[int, 1], prompt: str):
    await interaction.response.send_message(f"{minutes}分後にタイマーを設定したよ！⏰ 内容: 「{prompt}」についてお知らせするね。", ephemeral=True)
    print(f"タイマー設定: {minutes}分後, '{prompt}', User: {interaction.user}, Ch: {interaction.channel}")
    asyncio.create_task(execute_timer(interaction.channel, minutes, prompt, interaction.user))

# --- タイマー実行関数 ---
# (変更なし)
async def execute_timer(channel: discord.TextChannel, minutes: int, prompt: str, user: discord.User):
    await asyncio.sleep(minutes * 60)
    print(f"タイマー実行: {minutes}分経過, '{prompt}', User: {user}, Ch: {channel.name}")
    try:
        timer_execution_prompt = f"「{prompt}」というリマインダーの時間になりました。ユーザー ({user.display_name}) に向けて、時間になったことを知らせる、あなたのペルソナに合ったメッセージを生成してください。"
        async with channel.typing():
            _used_model, response_text = await generate_gemini_response([{'text': timer_execution_prompt}], use_primary_model=True)

        is_error_response = response_text is None or response_text.startswith(("ごめん", "ふえぇ"))
        if not is_error_response:
             mention = user.mention
             full_message = f"⏰ {mention} {minutes}分前に設定されたタイマーの時間だよ！\n\n{response_text}"
             for i in range(0, len(full_message), 1990): await channel.send(full_message[i:i+1990])
        else:
             error_msg = response_text if response_text else "（メッセージ生成に失敗しちゃった…）"
             await channel.send(f"⏰ {user.mention} {minutes}分前に設定されたタイマーの時間だよ！\n\n{error_msg}")
    except discord.Forbidden:
        print(f"エラー: タイマーメッセージ送信権限なし (Ch: {channel.name})")
    except Exception as e:
        print(f"エラー: タイマー実行中にエラー発生: {e}")
        try:
            await channel.send(f"⏰ {user.mention} タイマーの時間だけど、お知らせメッセージの生成中にエラーが起きたみたい…\nリマインダー内容: 「{prompt}」")
        except Exception as send_e:
             print(f"エラー: タイマーのエラー通知送信にも失敗: {send_e}")


# --- 投票機能 ---
# (変更なし)
class PollView(View):
    def __init__(self, question: str, options: list[str], author: discord.User):
        super().__init__(timeout=None)
        self.question = question
        self.options = options
        self.author = author
        self.votes = {option: set() for option in options}
        self.closed = False
        self.message = None

        option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
        for i, option in enumerate(options):
            button = Button(label=f"{option} (0票)", style=discord.ButtonStyle.secondary, custom_id=f"poll_option_{i}", emoji=option_emojis[i])
            button.callback = self.button_callback
            self.add_item(button)

        close_button = Button(label="投票を締め切る", style=discord.ButtonStyle.danger, custom_id="poll_close")
        close_button.callback = self.close_callback
        self.add_item(close_button)

    async def update_embed(self, interaction: discord.Interaction):
        if not self.message: self.message = interaction.message

        embed = self.message.embeds[0]
        option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
        options_text = "".join(f"{option_emojis[i]} {option} - {len(self.votes[option])}票\n" for i, option in enumerate(self.options))
        if embed.fields:
             embed.set_field_at(0, name="選択肢" if not self.closed else "最終結果", value=options_text, inline=False)
        else:
             embed.add_field(name="選択肢" if not self.closed else "最終結果", value=options_text, inline=False)

        for i, option in enumerate(self.options):
            button = discord.utils.get(self.children, custom_id=f"poll_option_{i}")
            if button:
                button.label = f"{option} ({len(self.votes[option])}票)"
                button.disabled = self.closed

        close_button = discord.utils.get(self.children, custom_id="poll_close")
        if close_button:
             close_button.disabled = self.closed
             if self.closed: close_button.label = "締め切り済み"

        try:
             await self.message.edit(embed=embed, view=self)
        except discord.NotFound: print("投票Embed更新エラー: メッセージが見つかりません。")
        except discord.Forbidden: print("投票Embed更新エラー: メッセージ編集権限がありません。")
        except Exception as e: print(f"投票Embed更新中にエラー: {e}")


    async def button_callback(self, interaction: discord.Interaction):
        if self.closed:
            await interaction.response.send_message("ごめんね、この投票はもう締め切られちゃったんだ…", ephemeral=True)
            return

        custom_id = interaction.data['custom_id']
        option_index = int(custom_id.split('_')[-1])
        selected_option = self.options[option_index]
        user_id = interaction.user.id

        voted_message = ""
        removed_vote = False
        for option, voters in self.votes.items():
            if user_id in voters:
                if option == selected_option:
                    voters.remove(user_id)
                    voted_message = f"「{selected_option}」への投票を取り消したよ。"
                    removed_vote = True
                else:
                    voters.remove(user_id)
                    self.votes[selected_option].add(user_id)
                    voted_message = f"投票を「{selected_option}」に変更したよ！"
                break
        else:
            if not removed_vote:
                self.votes[selected_option].add(user_id)
                voted_message = f"「{selected_option}」に投票したよ！ ありがとう！"

        await interaction.response.send_message(voted_message, ephemeral=True)
        await self.update_embed(interaction)


    async def close_callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.author.id:
            await interaction.response.send_message("ごめんね、投票を締め切れるのは作った人だけなんだ🙏", ephemeral=True)
            return

        if self.closed:
             await interaction.response.send_message("この投票はもう締め切られてるよ！", ephemeral=True)
             return

        self.closed = True
        await interaction.response.defer()

        if not self.message: self.message = interaction.message
        embed = self.message.embeds[0]
        embed.title = f"📊 投票結果！: {self.question}"
        embed.description = "投票ありがとう！ 結果はこうなったよ！"
        embed.color = discord.Color.red()

        await self.update_embed(interaction)
        print(f"投票締め切り: '{self.question}', User: {interaction.user}")


@client.tree.command(name="poll", description="投票を作成します（選択肢は2～10個）")
@app_commands.describe(
    question="投票の質問内容",
    option1="選択肢1", option2="選択肢2",
    option3="選択肢3", option4="選択肢4", option5="選択肢5",
    option6="選択肢6", option7="選択肢7", option8="選択肢8",
    option9="選択肢9", option10="選択肢10"
)
async def poll(interaction: discord.Interaction,
             question: str,
             option1: str, option2: str,
             option3: Optional[str] = None, option4: Optional[str] = None, option5: Optional[str] = None,
             option6: Optional[str] = None, option7: Optional[str] = None, option8: Optional[str] = None,
             option9: Optional[str] = None, option10: Optional[str] = None):
    options = [opt for opt in [option1, option2, option3, option4, option5, option6, option7, option8, option9, option10] if opt is not None]

    if len(options) < 2:
        await interaction.response.send_message("選択肢は最低2つ必要だよ！", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    poll_prompt = f"以下の投票が作成されようとしています。この投票について、面白く、かつ投票を促すような短いコメントを一言、あなたのペルソナに沿って生成してください。\n\n質問: 「{question}」\n選択肢: {', '.join(options)}"
    _used_model, gemini_comment = await generate_gemini_response([{'text': poll_prompt}])
    comment = gemini_comment if gemini_comment and not gemini_comment.startswith(("ごめん", "ふえぇ")) else "みんな、下のボタンで投票してね！"

    embed = discord.Embed(title=f"📊 投票だよ！: {question}",
                          description=comment,
                          color=discord.Color.blue())

    view = PollView(question, options, interaction.user)

    option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
    options_text = "".join(f"{option_emojis[i]} {option} - 0票\n" for i, option in enumerate(options))
    embed.add_field(name="選択肢", value=options_text, inline=False)
    embed.set_footer(text=f"投票を作った人: {interaction.user.display_name}")

    try:
        poll_message = await interaction.followup.send(embed=embed, view=view)
        view.message = poll_message
        print(f"投票作成: '{question}', Options: {len(options)}, Comment: {comment[:30]}...")
    except discord.Forbidden:
        await interaction.followup.send("ふえぇ、メッセージを送るかインタラクションを作る権限がないみたい…", ephemeral=True)
    except Exception as e:
        print(f"投票作成エラー: {e}")
        await interaction.followup.send(f"投票を作ろうとしたらエラーになっちゃった…\n```\n{e}\n```", ephemeral=True)

# --- イベントハンドラ ---
# (変更なし)
@client.event
async def on_message(message):
    if message.author == client.user:
        return

# --- エラーハンドリング ---
# (変更なし)
@client.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandNotFound):
        # 応答済みかチェック
        if not interaction.response.is_done():
            await interaction.response.send_message("あれれ？そんなコマンド知らないなぁ…", ephemeral=True)
    elif isinstance(error, app_commands.CommandOnCooldown):
        if not interaction.response.is_done():
            await interaction.response.send_message(f"このコマンドはクールダウン中です。{error.retry_after:.2f}秒後に試してね。", ephemeral=True)
    elif isinstance(error, app_commands.MissingPermissions):
        if not interaction.response.is_done():
            await interaction.response.send_message("ごめんね、このコマンドを実行する権限がないみたい…🙏", ephemeral=True)
    elif isinstance(error, app_commands.BotMissingPermissions):
        if not interaction.response.is_done():
            await interaction.response.send_message("ふえぇ、僕に必要な権限がないみたい… サーバー管理者に確認してみてね。", ephemeral=True)
    elif isinstance(error, app_commands.CheckFailure):
         if not interaction.response.is_done():
             await interaction.response.send_message("ごめんなさい、このコマンドはここでは使えないみたい。", ephemeral=True)
    elif isinstance(error, app_commands.TransformerError):
         if not interaction.response.is_done():
             await interaction.response.send_message(f"コマンドの引数の使い方がちょっと違うみたい。\n`{str(error)}`", ephemeral=True)
    else:
        print(f"スラッシュコマンドエラー発生: {error}")
        error_message = f"コマンド実行中に予期せぬエラーが発生しました。\n```\n{type(error).__name__}: {error}\n```"
        try:
             if interaction.response.is_done():
                  # followup.send は ephemeral=True をサポートしている
                  await interaction.followup.send(error_message, ephemeral=True)
             else:
                  await interaction.response.send_message(error_message, ephemeral=True)
        except Exception as e:
             print(f"エラーハンドラでのメッセージ送信中にさらにエラー: {e}")


# --- BOT起動 ---
# (変更なし)
if __name__ == "__main__":
    try: import aiofiles
    except ImportError: print("CRITICAL: 'aiofiles' がインストールされていません。`pip install aiofiles` を実行してください。"); exit()
    try: import discord
    except ImportError: print("CRITICAL: 'discord.py' がインストールされていません。`pip install -U discord.py` を実行してください。"); exit()
    try: import google.generativeai
    except ImportError: print("CRITICAL: 'google-generativeai' がインストールされていません。`pip install google-generativeai` を実行してください。"); exit()
    try: import dotenv
    except ImportError: print("CRITICAL: 'python-dotenv' がインストールされていません。`pip install python-dotenv` を実行してください。"); exit()

    missing_vars = []
    if not DISCORD_TOKEN: missing_vars.append("DISCORD_TOKEN")
    if not GEMINI_API_KEY: missing_vars.append("GEMINI_API_KEY")
    if not PRIMARY_MODEL_NAME: missing_vars.append("PRIMARY_GEMINI_MODEL")
    if not SECONDARY_MODEL_NAME: missing_vars.append("SECONDARY_GEMINI_MODEL")
    if not LOWLOAD_MODEL_NAME: missing_vars.append("LOWLOAD_GEMINI_MODEL")

    if missing_vars:
         print(f"CRITICAL: 以下の環境変数が .env ファイルに設定されていません: {', '.join(missing_vars)}")
    else:
        try:
            print("BOT起動中...")
            client.run(DISCORD_TOKEN)
        except discord.LoginFailure:
            print("CRITICAL: 不正なDiscordトークンです。 .envファイルを確認してください。")
        except discord.PrivilegedIntentsRequired:
            print("CRITICAL: 必要な特権インテント（Message Contentなど）が無効になっています。Discord Developer PortalでBOTの設定を確認してください。")
        except Exception as e:
            print(f"CRITICAL: BOT実行中に予期せぬエラーが発生しました: {e}")