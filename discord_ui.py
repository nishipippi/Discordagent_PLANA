# discord_ui.py
# (Discord UI要素: ボタンビュー、思考中メッセージ)

import discord
import asyncio
from typing import List, Optional, Dict, Any, Union, Literal # Union, Literalをインポート

import config
import bot_constants
import llm_manager
import cache_manager # ボタンコールバックでのキャッシュ操作用

# llm_provider モジュールから必要なエラータイプをインポート
from llm_provider import ERROR_TYPE_UNKNOWN # <-- ERROR_TYPE_UNKNOWN をインポート

# --- 思考中メッセージ管理 ---
_thinking_message: Optional[discord.Message] = None
_last_thinking_channel_id: Optional[int] = None # 最後に送信したチャンネルを記憶

# channel 引数の型ヒントを修正
# Literal[discord.utils.MISSING] がエラーになる場合、より汎用的な型 (object) で代用
# 実行時の `is discord.utils.MISSING` チェックはそのまま維持
async def update_thinking_message(channel: Union[discord.TextChannel, object], message_content: str):
    """思考中メッセージを更新または新規作成する"""
    global _thinking_message, _last_thinking_channel_id

    target_channel: Optional[discord.TextChannel] = None

    # channel が discord.TextChannel オブジェクトの場合
    if isinstance(channel, discord.TextChannel):
        target_channel = channel
        _last_thinking_channel_id = channel.id # チャンネルが指定されたら記憶

    # channel が discord.utils.MISSING オブジェクトの場合
    elif channel is discord.utils.MISSING:
        # channel省略時は既存メッセージのチャンネルを使う
        if _thinking_message:
            target_channel = _thinking_message.channel
        # _last_thinking_channel_id がある場合、本来はクライアントからチャンネルを取得したいが、
        # この関数単体ではクライアント情報がない。既存メッセージもなければ新規送信はできない。
        elif _last_thinking_channel_id:
             print(f"Warning: Cannot send new thinking message to channel {_last_thinking_channel_id} without channel object or existing message.")
             return # チャンネルオブジェクトがないので処理中断
        else:
            # メッセージもチャンネルIDも不明
            print("Warning: Cannot update thinking message without channel context.")
            return # 処理中断
    else:
        print("Error: Invalid channel information type for update_thinking_message.")
        return # 処理中断

    # target_channelがNoneでなければ処理続行
    if target_channel is None:
         print("Error: Internal logic error, target_channel is None after checks.")
         return # 念のため

    try:
        # 既存メッセージがあり、かつ同じチャンネルであれば編集
        if _thinking_message and _thinking_message.channel.id == target_channel.id:
            await _thinking_message.edit(content=message_content)
        else:
            # チャンネルが異なるか、メッセージが存在しない場合は新規作成
            # 古いメッセージがあれば削除を試みる (チャンネルが変わった場合など)
            await delete_thinking_message()
            _thinking_message = await target_channel.send(message_content)
            _last_thinking_channel_id = target_channel.id # 新規作成時もチャンネルIDを記憶
    except (discord.NotFound, discord.Forbidden):
         print(f"Warning: Failed to edit/send thinking message in channel {target_channel.id}. Resetting.")
         _thinking_message = None
         _last_thinking_channel_id = None # メッセージもチャンネル情報も失われたらリセット
    except Exception as e:
         print(f"Error updating thinking message in channel {target_channel.id}: {e}")
         _thinking_message = None
         _last_thinking_channel_id = None # エラー時もリセット


async def delete_thinking_message():
    """思考中メッセージを削除する"""
    global _thinking_message
    if _thinking_message:
        try:
            await _thinking_message.delete()
            # print("Thinking message deleted.") # ログ抑制
        except (discord.NotFound, discord.Forbidden):
            # print("Warning: Thinking message already deleted or cannot be deleted.") # ログ抑制
            pass # 削除済みや権限不足はエラーではない
        except Exception as e:
            print(f"Error deleting thinking message: {e}")
        finally:
            _thinking_message = None
            # _last_thinking_channel_id は削除時にはクリアしない方が良いかもしれない
            # すぐに次の thinking message が update で呼ばれる可能性があるため
            # ただし、上記 update ロジックで明確に target_channel が取得できない場合は
            # 新規送信ができないようにしているので、ここでのクリアは必須ではないかも。
            # 一貫性のため、メッセージが無くなったらチャンネル情報もリセットする方が分かりやすいか？
            # 一旦保留（今回の修正案では削除時に _last_thinking_channel_id は触らない）


# --- 追跡質問ボタン ---
class FollowUpView(discord.ui.View):
    """追跡質問ボタンを表示するView"""
    def __init__(self, original_message: Union[discord.Message, discord.WebhookMessage], follow_up_prompts: List[str]):
        super().__init__(timeout=config.FOLLOW_UP_BUTTON_TIMEOUT)
        # WebhookMessage も格納できるように型ヒントを Union に変更
        self.original_message: Union[discord.Message, discord.WebhookMessage] = original_message
        self.follow_up_prompts = follow_up_prompts

        for i, prompt_text in enumerate(follow_up_prompts):
            # ラベルは最大80文字
            button_label = prompt_text[:80]
            button = discord.ui.Button(label=button_label, style=discord.ButtonStyle.secondary, custom_id=f"follow_up_{i}")
            button.callback = self.button_callback # コールバック関数を設定
            self.add_item(button)

    async def on_timeout(self):
        """タイムアウトしたらボタンを無効化してViewを削除"""
        for item in self.children:
            if isinstance(item, discord.ui.Button): item.disabled = True
        try:
             # 編集対象のメッセージが存在するか確認してから編集
             if self.original_message:
                  # WebhookMessage にも edit があることを期待
                  await self.original_message.edit(view=None)
             # print(f"ボタンタイムアウト ({self.original_message.id})。ボタン削除。") # ログ抑制
        except (discord.NotFound, discord.Forbidden): pass # メッセージが見つからない or 権限がない場合は無視
        except Exception as e: print(f"ボタンタイムアウト後のメッセージ編集中にエラー: {e}")
        self.stop() # View自体を停止

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """誰でもボタンを押せるようにする"""
        return True # 誰でも押せる

    async def button_callback(self, interaction: discord.Interaction):
        """ボタンが押されたときの処理"""
        llm_handler = llm_manager.get_current_provider()
        if not llm_handler:
            await interaction.response.send_message(bot_constants.ERROR_MSG_INTERNAL + " (LLM Handler not available)", ephemeral=True)
            return

        # Thinking表示 (defer) - ボタン押下時はdeferが必要
        await interaction.response.defer(thinking=True, ephemeral=False)

        # 押されたボタンの特定
        button_label_full_prompt = "" # 元のプロンプト全文
        custom_id = interaction.data.get("custom_id", "") if interaction.data else ""
        if custom_id.startswith("follow_up_"):
            try:
                index = int(custom_id.split("_")[-1])
                if 0 <= index < len(self.follow_up_prompts):
                    button_label_full_prompt = self.follow_up_prompts[index] # 候補リストから元のプロンプトを取得
            except (ValueError, IndexError): pass

        if not button_label_full_prompt:
            await interaction.followup.send(bot_constants.ERROR_MSG_BUTTON_ERROR, ephemeral=True)
            return

        provider_name = llm_manager.get_current_provider_name()
        print(f"追跡質問ボタン押下: '{button_label_full_prompt}' by {interaction.user.display_name} (Provider: {provider_name})")

        channel = interaction.channel # interaction からチャンネルを取得
        channel_id = interaction.channel_id

        if not channel_id or not channel or not isinstance(channel, discord.TextChannel):
            print(f"Error: Cannot get channel info from interaction {interaction.id}.")
            await interaction.followup.send(bot_constants.ERROR_MSG_CHANNEL_ERROR, ephemeral=True)
            return

        # 元のメッセージからボタンを削除 (ユーザーが複数回押せないように)
        try:
             if self.original_message:
                  await self.original_message.edit(view=None)
        except (discord.NotFound, discord.Forbidden): pass # メッセージが見つからない or 権限がない場合は無視
        except Exception as e: print(f"追跡質問応答前のボタン削除エラー: {e}")
        self.stop() # Viewを停止

        # --- 応答生成処理 ---
        # ボタンコールバックでは Deep Cache と履歴を使用
        chat_history = await cache_manager.load_cache(channel_id)
        deep_cache_summary = await cache_manager.load_deep_cache(channel_id)
        user_entry_parts = [{'text': button_label_full_prompt}] # ボタンの全文をユーザー入力とする

        try:
            # LLM応答生成 (llm_manager経由)
            # ボタンからの応答なので履歴とDeep Cacheを考慮
            used_model_name, response_text_raw = await llm_manager.generate_response(
                content_parts=user_entry_parts,
                chat_history=chat_history,
                deep_cache_summary=deep_cache_summary
            )
            response_text = str(response_text_raw) if response_text_raw else ""
            print(f"Button callback LLM ({provider_name} - {used_model_name}) response received.")

            sent_followup_message: Optional[discord.WebhookMessage] = None # follow.send は WebhookMessage を返す
            is_error_response = llm_manager.is_error_message(response_text) # エラー判定を先に行う

            if response_text:
                # 応答を送信 (interaction.followup.send)
                # followup.send は 2000文字制限に注意
                response_to_send = response_text[:2000]
                if len(response_text) > 2000:
                     print("Warning: Follow-up response exceeds 2000 chars, truncated for followup.send.")

                # 応答送信 (2000文字以上の場合の考慮)
                # followup.send は最初の応答にしか使えないので、長い場合は channel.send に切り替える
                # ただし、thinking表示後の応答は followup.send が適切
                # 2000文字超える場合は、最初の2000文字を followup.send で送り、残りを channel.send で送るのが妥当
                if not is_error_response and len(response_text) > 2000:
                     print(f"Response text length ({len(response_text)}) exceeds 2000. Sending in chunks via channel.send.")
                     response_chunks = [response_text[i:i+1990] for i in range(0, len(response_text), 1990)]
                     first_chunk = True
                     for chunk in response_chunks:
                          if first_chunk:
                              # interaction.followup.send は defer 後に最初の応答として使う
                              sent_followup_message = await interaction.followup.send(chunk)
                              first_chunk = False
                          else:
                              # 2通目以降はチャンネルに直接送信
                              # WebhookMessage ではなく discord.Message になるが、
                              # 追跡ボタンは最初のメッセージにだけ追加すれば十分なので問題ない
                              await channel.send(chunk)
                          await asyncio.sleep(0.5) # 連投制限対策
                else:
                     # 2000文字以下またはエラーメッセージの場合は followup.send で一括送信
                     try:
                          sent_followup_message = await interaction.followup.send(response_text[:2000])
                     except discord.HTTPException as e:
                           print(f"Error sending followup response: {e}")
                           # エラーメッセージの送信も失敗したらログだけ
                           pass


                if not is_error_response:
                    # 正常応答の場合、キャッシュ更新
                    user_entry_parts_for_cache = user_entry_parts # ボタン応答の場合はテキストのみ
                    current_history = chat_history + [{'role': 'user', 'parts': user_entry_parts_for_cache}]
                    if response_text: # response_text が None でないことを確認
                        current_history.append({'role': 'model', 'parts': [{'text': response_text}]}) # 全文を保存
                    await cache_manager.save_cache(channel_id, current_history)
                    print("Cache updated after button callback.")

                    # 応答メッセージに対してさらにボタンを追加 (非同期で実行)
                    # WebhookMessage (sent_followup_message) に View を追加
                    if sent_followup_message:
                         asyncio.create_task(generate_and_add_followup_buttons(sent_followup_message, channel_id))
                    else:
                         print("Warning: Failed to get sent message object, cannot add further buttons.")
            else:
                # 応答が空だった場合
                # ERROR_TYPE_UNKNOWN は llm_provider からインポート済み
                err_msg = llm_handler.format_error_message(ERROR_TYPE_UNKNOWN, "Empty response from API.") if llm_handler else bot_constants.ERROR_MSG_GEMINI_UNKNOWN
                # followup.send は一度しか使えないため、エラーメッセージも followup で送る
                try:
                     await interaction.followup.send(err_msg[:2000])
                except Exception as send_err:
                     print(f"Error sending empty response error message: {send_err}")


        except Exception as e:
             print(f"エラー: ボタンコールバック中の応答生成または送信に失敗: {e}")
             try:
                 err_msg = bot_constants.ERROR_MSG_INTERNAL + f" (詳細: {str(e)[:100]})"
                 # followup.send は一度しか使えないため、例外時のエラーも followup で送る
                 await interaction.followup.send(err_msg[:2000], ephemeral=True) # エラー時はephemeralにする方が良いかも
             except Exception as send_err:
                 print(f"エラー: 例外処理中のエラーメッセージ送信に失敗: {send_err}")


async def generate_and_add_followup_buttons(
        message_to_edit: Union[discord.Message, discord.WebhookMessage], # 通常メッセージとWebhookメッセージ両対応
        channel_id: int):
    """追跡質問ボタンを生成し、メッセージに追加する"""
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        print("追跡質問ボタン生成スキップ: LLMハンドラー未初期化。")
        return

    lowload_model_name = llm_handler.get_model_name('lowload')
    provider_name = llm_manager.get_current_provider_name()
    if not lowload_model_name:
        # print(f"追跡質問ボタン生成スキップ: 低負荷モデル利用不可 ({provider_name})。") # 頻繁に出るのでログ抑制
        return

    # print(f"追跡質問ボタン生成試行 (Channel: {channel_id}, Model: {lowload_model_name})...") # ログ抑制

    chat_history = await cache_manager.load_cache(channel_id)
    if not chat_history:
        # print("追跡質問ボタン生成スキップ: キャッシュ履歴なし。") # ログ抑制
        return

    # 直近の履歴を整形 (例: 2往復分 = 4エントリ)
    recent_history_entries = chat_history[-4:]
    # _format_history_for_prompt は cache_manager モジュールにあるため、cache_manager. をつける
    recent_history_text = cache_manager._format_history_for_prompt(recent_history_entries)

    if not recent_history_text.strip():
        # print("追跡質問ボタン生成スキップ: 履歴テキスト空。") # ログ抑制
        return

    # プロンプト生成とLLM呼び出し
    button_prompt = config.FOLLOW_UP_PROMPT.format(
        max_buttons=config.MAX_FOLLOW_UP_BUTTONS,
        recent_history_text=recent_history_text
    )
    follow_up_suggestions_raw = await llm_manager.generate_lowload_response(button_prompt)
    follow_up_suggestions = str(follow_up_suggestions_raw) if follow_up_suggestions_raw else ""

    # 応答をパース
    if follow_up_suggestions and "提案なし" not in follow_up_suggestions.lower() and not llm_manager.is_error_message(follow_up_suggestions):
        follow_up_prompts = [line.strip() for line in follow_up_suggestions.split('\n') if line.strip()][:config.MAX_FOLLOW_UP_BUTTONS]
        follow_up_prompts = [p for p in follow_up_prompts if len(p) >= 3] # 短すぎる候補を除外

        if follow_up_prompts:
             # print(f"生成された追跡質問候補: {follow_up_prompts}") # ログ抑制
             try:
                 # WebhookMessageの場合でもeditできることを期待
                 view = FollowUpView(original_message=message_to_edit, follow_up_prompts=follow_up_prompts)
                 await message_to_edit.edit(view=view)
                 # print("追跡質問ボタンをメッセージに追加しました。") # ログ抑制
             except (discord.NotFound, discord.Forbidden):
                 # print(f"警告: ボタン追加対象メッセージ({message_to_edit.id})が見つからないか編集権限がありません。") # ログ抑制
                 pass
             except AttributeError:
                  print(f"警告: メッセージオブジェクト ({type(message_to_edit)}) が edit メソッドをサポートしていません。")
             except Exception as e:
                 print(f"エラー: 追跡質問ボタンのメッセージへの追加中にエラー: {e}")
        # else: print("低負荷モデルから有効な追跡質問候補が得られませんでした。") # ログ抑制
    # else: # 提案なし or エラー (ログ抑制)