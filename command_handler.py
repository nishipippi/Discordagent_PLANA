# command_handler.py
# (Discordコマンドの処理、メンション応答ロジック)

import discord
import re
import asyncio
import mimetypes
import io # PDF処理用に追加
from typing import List, Dict, Any, Optional, Union

# PDF処理ライブラリをインポート (requirements.txt に pypdf2 を追加してください)
try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError # PyPDF2 v3+
except ImportError:
    PdfReader = None # ライブラリがない場合はNoneにしておく
    PdfReadError = Exception # 適当な例外クラス
    print("警告: PyPDF2 がインストールされていないため、PDFファイルの処理はスキップされます。")
    print("`pip install pypdf2` を実行してください。")


import config
import bot_constants
import llm_manager
import cache_manager
import discord_ui # ボタン生成用
from llm_provider import ERROR_TYPE_UNKNOWN, ERROR_TYPE_INTERNAL # エラータイプ定数

# --- PDFテキスト抽出関数 ---
async def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """PDFバイトデータからテキストを抽出する"""
    if PdfReader is None: # ライブラリがない場合
        return "[PDF処理不可 (ライブラリ未導入)]"

    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        num_pages = len(reader.pages)
        print(f"Extracting text from PDF ({num_pages} pages)...")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: # テキストが抽出できた場合のみ追加
                    text += page_text + "\n" # ページ間に改行を入れる
                # else: # 画像ベースのページなど
                #     print(f"  - Page {i+1}: No text extracted.")
            except Exception as page_e:
                print(f"Error extracting text from PDF page {i+1}: {page_e}")
                text += f"[ページ{i+1} 抽出エラー]\n"

        # PyPDF2は画像ベースのPDFからはテキストを抽出できないので、textが空になることがある
        if not text.strip():
             print("Warning: PDF text extraction resulted in empty string (possibly image-based PDF).")
             return "[PDF内容の抽出失敗 (テキスト情報なし)]" # LLMに失敗したことを伝える

        # 長すぎるテキストを制限する場合 (必要なら)
        # MAX_PDF_TEXT = 20000 # 例: 2万文字
        # if len(text) > MAX_PDF_TEXT:
        #     print(f"Warning: PDF text truncated to {MAX_PDF_TEXT} characters.")
        #     text = text[:MAX_PDF_TEXT] + "... (PDF text truncated)"
        print(f"PDF text extraction successful ({len(text)} chars).")
        return text.strip()
    except PdfReadError as pdf_err: # PyPDF2固有のエラーをキャッチ
        print(f"Error reading PDF (PdfReadError): {pdf_err}")
        return "[PDF読み込みエラー (ファイル破損または非対応形式)]"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラーログ
        return "[PDF内容の抽出中に予期せぬエラー]" # LLMにエラーを伝える

# --- タイマー実行 ---
async def execute_timer(channel: discord.TextChannel, minutes: int, prompt: str, author: discord.User):
    """タイマーを実行し、LLMによる補足メッセージ付きで通知する"""
    await asyncio.sleep(minutes * 60)

    llm_handler = llm_manager.get_current_provider()
    provider_name = llm_manager.get_current_provider_name()

    if not llm_handler:
        print(f"タイマー実行エラー: LLMハンドラーが利用できません ({channel.name}, {prompt})")
        try:
            await channel.send(f"{author.mention} タイマー「{prompt[:100]}...」の通知時刻ですが、内部エラーで補足メッセージを生成できませんでした。")
        except discord.HTTPException as e:
            print(f"Error sending timer error message: {e}")
        return

    print(f"タイマー実行: {minutes}分経過, '{prompt[:50]}...', Ch: {channel.name}, Author: {author.display_name}, Provider: {provider_name}")

    async with channel.typing():
        mention = author.mention
        base_message = f"{mention} 指定時刻です。\nタイマーの内容: 「{prompt}」"

        # 補足メッセージ生成
        timer_execution_prompt = f"「{prompt}」というリマインダーの指定時刻になりました。ユーザー ({author.display_name}) に向けて、簡潔な補足メッセージを生成してください。（現在の状況や時間帯なども少し考慮すると良いでしょう）"
        response_text = ""
        try:
            _used_model, response_text_raw = await llm_manager.generate_response(
                content_parts=[{'text': timer_execution_prompt}], chat_history=None, deep_cache_summary=None
            )
            response_text = str(response_text_raw) if response_text_raw else ""
        except Exception as e:
             print(f"Error generating timer follow-up message: {e}")
             response_text = llm_handler.format_error_message(ERROR_TYPE_INTERNAL, f"Timer generation failed: {e}")

        full_message = base_message
        if response_text and not llm_manager.is_error_message(response_text):
            full_message += f"\n\n{response_text}"
        elif response_text: # エラーメッセージの場合
             print(f"タイマー補足生成失敗: {response_text}")
             full_message += f"\n\n({response_text[:150]})" # 短縮して表示

        # メッセージ送信 (2000文字制限考慮)
        try:
            if len(full_message) > 2000:
                 # 2000文字を超える場合は分割送信
                 await channel.send(full_message[:1990])
                 await channel.send(full_message[1990:3980]) # 2通目まで
            else:
                 await channel.send(full_message)
        except discord.HTTPException as e:
            print(f"Error sending timer execution message: {e}")


# --- コマンド処理 ---
async def handle_command(message: discord.Message):
    """メッセージ内容を解析し、コマンドを実行する
    注意: 検索コマンド (!src, !dsrc) は bot.py の on_message で処理される
    """
    if not message.content: return False # コマンドなし

    content_lower = message.content.lower().strip()
    channel_id = message.channel.id

    # --- プロバイダー切り替えコマンド ---
    target_provider_name: Optional[str] = None
    if content_lower == '!gemini':
        target_provider_name = 'GEMINI'
    elif content_lower == '!mistral':
        target_provider_name = 'MISTRAL'

    if target_provider_name:
        async with message.channel.typing():
             success, response_msg = await llm_manager.switch_provider(target_provider_name)
             await message.reply(response_msg, mention_author=False)
             if success:
                  # bot.py側で presence 更新済み
                  pass
        return True # コマンド処理完了

    # --- キャッシュ操作コマンド ---
    if content_lower == '!csum':
        async with message.channel.typing():
             success, response_msg = await cache_manager.summarize_deep_cache(channel_id)
             await message.reply(response_msg, mention_author=False)
        return True
    elif content_lower == '!cclear':
        async with message.channel.typing():
             print(f"Deep Cache クリア実行 (!cclear, Channel: {channel_id})...")
             await cache_manager.save_deep_cache(channel_id, None) # Noneを保存してクリア
             print(f"Deep Cache クリア完了 (Channel: {channel_id})。")
             await message.reply("長期記憶(Deep Cache)を初期化しました。", mention_author=False)
        return True

    # --- タイマーコマンド ---
    if content_lower.startswith('!timer '):
        match = re.match(r'!timer\s+(\d+)\s*(分|分後|minute|minutes)\s*(.*)', message.content, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                minutes = int(match.group(1))
                timer_prompt = match.group(3).strip()
                if not timer_prompt:
                    await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 内容を指定してください。", mention_author=False); return True
                if not (1 <= minutes <= 1440): # 1分以上24時間以下
                     await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 時間は1分以上1440分以下で指定してください。", mention_author=False); return True

                provider_name = llm_manager.get_current_provider_name()
                await message.channel.send(f"{minutes}分後にタイマーを設定しました ({provider_name}が通知します)。\n内容: 「{timer_prompt[:100]}...」")
                print(f"タイマー設定: {minutes}分後, '{timer_prompt[:50]}...', Ch: {message.channel.name}, Author: {message.author.display_name}, Provider: {provider_name}")
                # タイマー実行を非同期タスクとしてスケジュール
                asyncio.create_task(execute_timer(message.channel, minutes, timer_prompt, message.author)) # type: ignore
            except ValueError:
                 await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 時間は半角数字で指定してください。", mention_author=False)
        else:
            await message.reply(bot_constants.ERROR_MSG_TIMER_INVALID + " 例: `!timer 10分 会議リマインダー`", mention_author=False)
        return True # コマンド処理完了

    # --- 投票コマンド ---
    if content_lower.startswith('!poll '):
        args = message.content.split(' ', 1)
        if len(args) < 2 or not args[1].strip():
            await message.reply(bot_constants.ERROR_MSG_POLL_INVALID + " 内容を指定してください。", mention_author=False); return True
        poll_content = args[1].strip()
        # ダブルクォートで囲まれた部分を優先的に抽出
        parts = re.findall(r'"([^"]*)"|\S+', poll_content)
        question = ""
        options = []
        if len(parts) > 0:
            question = parts[0] # 最初の要素（クォート除去済み or 最初の単語）
            options = [p.strip() for p in parts[1:] if p.strip()] # 残りをオプション

        if not question or not (2 <= len(options) <= 10):
            await message.reply(bot_constants.ERROR_MSG_POLL_INVALID + ' 例: `!poll "今日のランチは？" カレー ラーメン 定食`', mention_author=False); return True

        async with message.channel.typing():
            embed = discord.Embed(title=f"投票: {question}", description="以下から選択してください。", color=discord.Color.blue())
            option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
            options_text = "".join(f"{option_emojis[i]} {option}\n" for i, option in enumerate(options))
            embed.add_field(name="選択肢", value=options_text, inline=False)
            embed.set_footer(text=f"作成者: {message.author.display_name}")
            try:
                 poll_message = await message.channel.send(embed=embed)
                 for i in range(len(options)):
                     await poll_message.add_reaction(option_emojis[i])
                 print(f"投票作成: {question} by {message.author.display_name}")
            except discord.Forbidden:
                 await message.channel.send(bot_constants.ERROR_MSG_PERMISSION_DENIED + " (メッセージ送信/リアクション追加)")
            except Exception as e:
                 print(f"投票作成エラー: {e}"); await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " 投票作成失敗。")
        return True # コマンド処理完了

    return False # どのコマンドにも一致しなかった

# --- メンション応答処理 ---
async def handle_mention(message: discord.Message, client_user: discord.ClientUser):
    """メンションを受けた際の応答処理 (検索コマンドは除く)"""
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        # 通常、on_message側でチェックされるはずだが念のため
        print("Error: LLM Provider not available during mention handling.")
        await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not available)", mention_author=False)
        return

    channel_id = message.channel.id
    provider_name = llm_manager.get_current_provider_name()
    print(f"Mention received in channel {channel_id}. Processing with {provider_name}...")

    async with message.channel.typing():
        # 1. プロンプトと添付ファイルの準備
        mention_strings = [f'<@!{client_user.id}>', f'<@{client_user.id}>']
        text_content = message.content if message.content else ""
        for mention in mention_strings: text_content = text_content.replace(mention, '')
        text_content = text_content.strip()

        # !his フラグのチェック
        use_channel_history = False
        if '!his' in text_content.lower():
             # 単語として完全に一致する場合のみフラグを立てる (例: "!history" は対象外)
             if re.search(r'\b!his\b', text_content, re.IGNORECASE):
                 use_channel_history = True
                 text_content = re.sub(r'\b!his\b', '', text_content, flags=re.IGNORECASE).strip()
                 print("履歴参照フラグ (!his) 検出。キャッシュ無視。")

        # request_parts: LLM APIへの入力パーツリスト
        request_parts: List[Dict[str, Any]] = []
        # user_entry_parts_for_cache: キャッシュ保存用のユーザー入力パーツリスト
        user_entry_parts_for_cache: List[Dict[str, Any]] = []

        if text_content:
             request_parts.append({'text': text_content})
             user_entry_parts_for_cache.append({'text': text_content})

        # 添付ファイル処理
        file_error_occurred_once = False
        MAX_IMAGES = 5
        image_count = 0
        FILE_LIMIT_MB = 50
        processed_files_count = 0
        pdf_texts_for_cache: List[str] = [] # PDFから抽出したテキストを一時保存

        if message.attachments:
            print(f"{len(message.attachments)}個の添付ファイルを検出。")
            for attachment in message.attachments:
                # サイズチェック
                if attachment.size > FILE_LIMIT_MB * 1024 * 1024:
                    if not file_error_occurred_once: await message.channel.send(bot_constants.ERROR_MSG_FILE_SIZE_LIMIT + f" ({FILE_LIMIT_MB}MB超過)"); file_error_occurred_once = True
                    print(f"警告: 添付 '{attachment.filename}' サイズ超過 ({attachment.size / (1024*1024):.2f} MB)。スキップ。")
                    continue

                mime_type = attachment.content_type
                if mime_type is None: # Content-Typeが不明な場合、ファイル名から推測
                    mime_type, _ = mimetypes.guess_type(attachment.filename)
                    mime_type = mime_type or 'application/octet-stream' # 不明なら汎用バイナリ

                # --- 画像処理 ---
                if mime_type.startswith("image/"):
                    image_count += 1
                    if image_count > MAX_IMAGES:
                        if not file_error_occurred_once: await message.channel.send(bot_constants.ERROR_MSG_MAX_IMAGE_SIZE); file_error_occurred_once = True
                        print(f"警告: 画像数超過 ({image_count} > {MAX_IMAGES})。 '{attachment.filename}' をスキップ。")
                        continue
                    try:
                        file_bytes = await attachment.read()
                        # リクエスト用パーツ (inline_data形式)
                        request_parts.append({'inline_data': {'mime_type': mime_type, 'data': file_bytes}})
                        # キャッシュ保存用データ (inline_data形式、bytesを保持)
                        user_entry_parts_for_cache.append({'inline_data': {'mime_type': mime_type, 'data': file_bytes}})
                        processed_files_count += 1
                        print(f"添付 '{attachment.filename}' ({mime_type}) をリクエストとキャッシュ(予定)に追加。")
                    except discord.HTTPException as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_IMAGE_READ_FAIL} (Discordエラー)"); file_error_occurred_once = True
                        print(f"エラー: 添付 '{attachment.filename}' 読込失敗 (Discord HTTP): {e}")
                    except Exception as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_INTERNAL} (ファイル処理エラー)"); file_error_occurred_once = True
                        print(f"エラー: 添付 '{attachment.filename}' 処理中に予期せぬエラー: {e}")

                # --- PDF処理 ---
                elif mime_type == 'application/pdf':
                    print(f"Processing PDF attachment: {attachment.filename}")
                    try:
                        file_bytes = await attachment.read()
                        extracted_text = await extract_text_from_pdf(file_bytes)
                        if extracted_text:
                            # LLMへのリクエストには抽出テキストを含める
                            pdf_request_text = f"--- PDFファイル '{attachment.filename}' の内容 ---\n{extracted_text}\n--- PDFファイルここまで ---"
                            request_parts.append({'text': pdf_request_text})
                            # キャッシュ保存用に抽出テキストを一時保持
                            pdf_texts_for_cache.append(pdf_request_text)
                            processed_files_count += 1
                            print(f"添付 '{attachment.filename}' (PDF) のテキストをリクエストに追加。({len(extracted_text)} chars)")
                        else:
                            # 抽出失敗またはテキストなし
                            if not file_error_occurred_once: await message.channel.send(f"PDF '{attachment.filename}' からテキストを抽出できませんでした。"); file_error_occurred_once = True
                            print(f"警告: PDF '{attachment.filename}' からテキスト抽出失敗または内容空。スキップ。")
                    except discord.HTTPException as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_IMAGE_READ_FAIL} (Discordエラー)"); file_error_occurred_once = True
                        print(f"エラー: PDF添付 '{attachment.filename}' 読込失敗 (Discord HTTP): {e}")
                    except Exception as e:
                        if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_INTERNAL} (PDF処理エラー)"); file_error_occurred_once = True
                        print(f"エラー: PDF添付 '{attachment.filename}' 処理中に予期せぬエラー: {e}")

                # --- テキストファイル処理 ---
                elif mime_type.startswith('text/'):
                     try:
                         file_bytes = await attachment.read()
                         # テキストファイルはデコードして text として扱う
                         try:
                             # よく使われるエンコーディングを試す
                             detected_encoding = 'utf-8' # デフォルト
                             try: text_content_from_file = file_bytes.decode(detected_encoding)
                             except UnicodeDecodeError:
                                  try: detected_encoding = 'shift_jis'; text_content_from_file = file_bytes.decode(detected_encoding)
                                  except UnicodeDecodeError:
                                       try: detected_encoding = 'cp932'; text_content_from_file = file_bytes.decode(detected_encoding)
                                       except Exception: raise # これ以上は諦める
                             print(f"Decoded text file '{attachment.filename}' with {detected_encoding}.")
                         except Exception as decode_err:
                              print(f"Error decoding text file '{attachment.filename}': {decode_err}")
                              if not file_error_occurred_once: await message.channel.send(f"テキストファイル '{attachment.filename}' のデコードに失敗しました。"); file_error_occurred_once = True
                              continue # 次のファイルへ

                         # request_parts とキャッシュ用リストに追加
                         text_part_content = f"--- 添付テキストファイル '{attachment.filename}' の内容 ---\n{text_content_from_file}\n--- テキストファイルここまで ---"
                         request_parts.append({'text': text_part_content})
                         user_entry_parts_for_cache.append({'text': text_part_content})
                         processed_files_count += 1
                         print(f"添付 '{attachment.filename}' (テキスト) をリクエストとキャッシュ(予定)に追加。")

                     except discord.HTTPException as e:
                         if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_IMAGE_READ_FAIL} (Discordエラー)"); file_error_occurred_once = True
                         print(f"エラー: テキスト添付 '{attachment.filename}' 読込失敗 (Discord HTTP): {e}")
                     except Exception as e:
                         if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_INTERNAL} (ファイル処理エラー)"); file_error_occurred_once = True
                         print(f"エラー: テキスト添付 '{attachment.filename}' 処理中に予期せぬエラー: {e}")

                # --- その他の未対応ファイル ---
                else:
                    print(f"警告: 未対応MIMEタイプ '{mime_type}' ({attachment.filename})。スキップ。")
                    if not file_error_occurred_once: await message.channel.send(f"{bot_constants.ERROR_MSG_ATTACHMENT_UNSUPPORTED} ({mime_type})"); file_error_occurred_once = True
                    continue


        # PDFから抽出したテキストをキャッシュ保存用リストに追加
        for pdf_text in pdf_texts_for_cache:
             user_entry_parts_for_cache.append({'text': pdf_text})


        # 送信するテキストも有効な添付ファイルもない場合
        if not request_parts:
            print("応答可能なテキストコンテンツも有効な添付ファイルもありません。処理をスキップします。")
            # メンションのみの場合は何か返す
            if not text_content and not message.attachments:
                await message.reply("…呼びましたか？", mention_author=False)
            else:
                # エラーメッセージを送信済みでなければ、内容がない旨を伝える
                if not file_error_occurred_once:
                     await message.reply(bot_constants.ERROR_MSG_NO_CONTENT + " (テキストか対応ファイル[画像/PDF/Text]を送ってね！)", mention_author=False)
            return

        # 2. 履歴の準備
        chat_history: List[Dict[str, Any]] = []
        if use_channel_history:
            print(f"チャンネル履歴 ({config.HISTORY_LIMIT}件) 取得中...")
            try:
                # discord.py 2.0+ では async for を使用
                history_messages = [msg async for msg in message.channel.history(limit=config.HISTORY_LIMIT + 1)] # +1して自分を除く
                history_messages.reverse() # 古い順に
                history_messages = history_messages[:-1] # トリガーメッセージ（自分自身）を除く

                for msg in history_messages:
                    role = 'model' if msg.author == client_user else 'user'
                    msg_parts = []
                    txt = msg.content or ""
                    # 履歴内の添付ファイルはテキストで示す (簡略化)
                    if msg.attachments: txt += " " + " ".join([f"[{att.filename} 添付]" for att in msg.attachments])
                    if txt.strip(): msg_parts.append({'text': txt.strip()})

                    # 有効なパーツがある場合のみ履歴に追加
                    if msg_parts:
                        chat_history.append({'role': role, 'parts': msg_parts})
                print(f"チャンネル履歴から {len(chat_history)} 件整形完了。")
            except discord.Forbidden:
                await message.reply(bot_constants.ERROR_MSG_PERMISSION_DENIED + " (履歴読み取り権限がありません)", mention_author=False); return
            except Exception as e:
                await message.reply(bot_constants.ERROR_MSG_HISTORY_READ_FAIL, mention_author=False); print(f"エラー: チャンネル履歴取得中に予期せぬエラー: {e}"); return
        else:
            # キャッシュを使用
            print(f"チャンネル {channel_id} のキャッシュ読込中...")
            chat_history = await cache_manager.load_cache(channel_id)
            print(f"キャッシュから {len(chat_history)} 件の履歴を読み込みました。")

        # 3. Deep Cacheの準備
        deep_cache_summary = await cache_manager.load_deep_cache(channel_id)
        if deep_cache_summary: print("Deep Cache情報を読み込みました。")

        # 4. LLM API呼び出し (llm_manager経由)
        used_model_name, response_text_raw = await llm_manager.generate_response(
            content_parts=request_parts, # LLMには画像バイナリとテキスト(PDF含む)を渡す
            chat_history=chat_history,
            deep_cache_summary=deep_cache_summary
        )
        response_text = str(response_text_raw) if response_text_raw else ""
        print(f"LLM ({provider_name} - {used_model_name}) response received.")

        # 5. 応答送信
        sent_message: Optional[discord.Message] = None # 送信したメッセージオブジェクトを保持
        is_error_response = llm_manager.is_error_message(response_text) # エラー判定を先に行う

        if response_text:
            # エラーでない場合のみ分割送信を考慮
            if not is_error_response and len(response_text) > 2000:
                print(f"Response text length ({len(response_text)}) exceeds 2000. Sending in chunks.")
                response_chunks = [response_text[i:i+1990] for i in range(0, len(response_text), 1990)]
                first_chunk = True
                try:
                    for chunk in response_chunks:
                        if first_chunk:
                            sent_message = await message.reply(chunk, mention_author=False)
                            first_chunk = False
                        else:
                            # 2通目以降は通常の送信 (sent_messageは最初のメッセージを指す)
                            await message.channel.send(chunk)
                        await asyncio.sleep(0.5) # 連投制限対策
                except discord.HTTPException as e:
                     print(f"Error sending chunked response: {e}")
                     # 途中で失敗しても、最初のメッセージが送れていれば sent_message には値が入る
                     if not sent_message: # 最初の送信で失敗した場合
                          await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")
            else:
                # 2000文字以下またはエラーメッセージ
                try:
                    sent_message = await message.reply(response_text[:2000], mention_author=False) # 念のため制限
                except discord.HTTPException as e:
                     print(f"Error sending final response: {e}")
                     await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")

        else:
            # 応答が空だった場合
            err_msg = llm_handler.format_error_message(ERROR_TYPE_UNKNOWN, "Empty response from API.") if llm_handler else bot_constants.ERROR_MSG_GEMINI_UNKNOWN
            sent_message = await message.reply(err_msg, mention_author=False)

        # 6. キャッシュ更新 (エラーでなく、履歴モードでない場合)
        # user_entry_parts_for_cache には、元のテキストメッセージ、画像等のinline_data(bytes)、PDF等の抽出テキストが含まれる
        if not is_error_response and not use_channel_history and user_entry_parts_for_cache:
            # キャッシュに Deep Cache summary は含めない
            current_history = chat_history + [{'role': 'user', 'parts': user_entry_parts_for_cache}]
            if response_text: # response_text が None でないことを確認
                current_history.append({'role': 'model', 'parts': [{'text': response_text}]}) # 全文を保存
            await cache_manager.save_cache(channel_id, current_history)
            print("Cache updated.")
        elif not user_entry_parts_for_cache:
             print("Skipping cache update because user entry parts are empty.")


        # 7. 追跡質問ボタン生成 (エラーでなく、メッセージ送信成功時)
        if sent_message and not is_error_response:
             # 非同期でボタン生成・追加を実行
             asyncio.create_task(discord_ui.generate_and_add_followup_buttons(sent_message, channel_id))