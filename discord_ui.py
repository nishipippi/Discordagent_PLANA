# discord_ui.py
# (Discord UI要素: ボタンビュー、思考中メッセージ)

import discord
import asyncio
from typing import List, Optional, Dict, Any, Union # Unionを追加

import config
import bot_constants
import llm_manager
import cache_manager # ボタンコールバックでのキャッシュ操作用

# --- 思考中メッセージ管理 ---
_thinking_message: Optional[discord.Message] = None

async def update_thinking_message(channel: discord.TextChannel, message_content: str):
    """思考中メッセージを更新または新規作成する"""
    global _thinking_message
    try:
        if _thinking_message and _thinking_message.channel.id == channel.id:
            await _thinking_message.edit(content=message_content)
        else:
            _thinking_message = None # チャンネルが変わった場合などにリセット
            _thinking_message = await channel.send(message_content)
    except (discord.NotFound, discord.Forbidden):
         print(f"Warning: Failed to edit/send thinking message in channel {channel.id}. Resetting.")
         _thinking_message = None
    except Exception as e:
         print(f"Error updating thinking message: {e}")
         _thinking_message = None

async def delete_thinking_message():
    """思考中メッセージを削除する"""
    global _thinking_message
    if _thinking_message:
        try:
            await _thinking_message.delete()
            print("Thinking message deleted.")
        except (discord.NotFound, discord.Forbidden):
            print("Warning: Thinking message already deleted or cannot be deleted.")
        except Exception as e:
            print(f"Error deleting thinking message: {e}")
        finally:
            _thinking_message = None


# --- 追跡質問ボタン ---
class FollowUpView(discord.ui.View):
    """追跡質問ボタンを表示するView"""
    def __init__(self, original_message: discord.Message, follow_up_prompts: List[str]):
        super().__init__(timeout=config.FOLLOW_UP_BUTTON_TIMEOUT)
        self.original_message = original_message # ボタンを追加したボットのメッセージ
        self.follow_up_prompts = follow_up_prompts

        for i, prompt_text in enumerate(follow_up_prompts):
            button_label = prompt_text[:80] # Discordボタンラベル上限
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
                  await self.original_message.edit(view=None)
             # print(f"ボタンタイムアウト ({self.original_message.id})。ボタン削除。") # ログ抑制
        except (discord.NotFound, discord.Forbidden): pass # メッセージが見つからない or 権限がない場合は無視
        except Exception as e: print(f"ボタンタイムアウト後のメッセージ編集中にエラー: {e}")
        self.stop() # View自体を停止

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """誰でもボタンを押せるようにする"""
        return True

    async def button_callback(self, interaction: discord.Interaction):
        """ボタンが押されたときの処理"""
        llm_handler = llm_manager.get_current_provider()
        if not llm_handler:
            await interaction.response.send_message(bot_constants.ERROR_MSG_INTERNAL + " (LLM Handler not available)", ephemeral=True)
            return

        # Thinking表示 (defer)
        await interaction.response.defer(thinking=True, ephemeral=False)

        # 押されたボタンの特定
        button_label = ""
        custom_id = interaction.data.get("custom_id", "") if interaction.data else ""
        if custom_id.startswith("follow_up_"):
            try:
                index = int(custom_id.split("_")[-1])
                if 0 <= index < len(self.follow_up_prompts):
                    button_label = self.follow_up_prompts[index]
            except (ValueError, IndexError): pass

        if not button_label:
            await interaction.followup.send(bot_constants.ERROR_MSG_BUTTON_ERROR, ephemeral=True)
            return

        provider_name = llm_manager.get_current_provider_name()
        print(f"追跡質問ボタン押下: '{button_label}' by {interaction.user.display_name} (Provider: {provider_name})")

        channel_id = interaction.channel_id
        if not channel_id or not interaction.channel or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.followup.send(bot_constants.ERROR_MSG_CHANNEL_ERROR, ephemeral=True)
            return

        # 元のメッセージからボタンを削除
        try:
             if self.original_message:
                  await self.original_message.edit(view=None)
        except (discord.NotFound, discord.Forbidden): pass
        except Exception as e: print(f"追跡質問応答前のボタン削除エラー: {e}")
        self.stop() # Viewを停止

        # --- 応答生成処理 ---
        chat_history = await cache_manager.load_cache(channel_id)
        deep_cache_summary = await cache_manager.load_deep_cache(channel_id)
        user_entry_parts = [{'text': button_label}]

        try:
            # LLM応答生成 (llm_manager経由)
            used_model_name, response_text_raw = await llm_manager.generate_response(
                content_parts=user_entry_parts,
                chat_history=chat_history,
                deep_cache_summary=deep_cache_summary
            )
            response_text = str(response_text_raw) if response_text_raw else ""
            print(f"Button callback LLM ({provider_name} - {used_model_name}) response received.")

            sent_followup_message: Optional[discord.WebhookMessage] = None
            if response_text:
                is_error_response = llm_manager.is_error_message(response_text)

                # 応答を送信 (followup.send)
                response_to_send = response_text[:2000] # 2000文字制限
                if len(response_text) > 2000:
                    print("Warning: Follow-up response exceeds 2000 chars, truncated.")

                sent_followup_message = await interaction.followup.send(response_to_send)

                if not is_error_response:
                    # 正常応答の場合、キャッシュ更新
                    current_history = chat_history + [{'role': 'user', 'parts': user_entry_parts}]
                    current_history.append({'role': 'model', 'parts': [{'text': response_text}]}) # 全文を保存
                    await cache_manager.save_cache(channel_id, current_history)
                    print("Cache updated after button callback.")

                    # 応答メッセージに対してさらにボタンを追加 (非同期で実行)
                    if sent_followup_message:
                         # followup.sendはWebhookMessageを返す。editは可能なのでView追加もできるはず。
                         # WebhookMessageをdiscord.Messageとして扱う必要がある場合があるため、fetch_messageを使う方が確実かもしれない。
                         # ただし、パフォーマンスのため、まずは直接 edit を試みる。
                         asyncio.create_task(generate_and_add_followup_buttons(sent_followup_message, channel_id))
                    else:
                         print("Warning: Failed to get sent message object, cannot add further buttons.")
            else:
                # 空応答
                await interaction.followup.send(llm_handler.format_error_message(ERROR_TYPE_UNKNOWN, "Empty response received from API.") if llm_handler else bot_constants.ERROR_MSG_GEMINI_UNKNOWN)

        except Exception as e:
             print(f"エラー: ボタンコールバック中の応答生成または送信に失敗: {e}")
             try:
                 err_msg = bot_constants.ERROR_MSG_INTERNAL + f" (詳細: {str(e)[:100]})"
                 await interaction.followup.send(err_msg[:2000], ephemeral=True)
             except Exception as send_err:
                 print(f"エラー: エラーメッセージの送信にも失敗: {send_err}")


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
        print(f"追跡質問ボタン生成スキップ: 低負荷モデル利用不可 ({provider_name})。")
        return

    print(f"追跡質問ボタン生成試行 (Channel: {channel_id}, Model: {lowload_model_name})...")

    chat_history = await cache_manager.load_cache(channel_id)
    if not chat_history:
        print("追跡質問ボタン生成スキップ: キャッシュ履歴なし。")
        return

    # 直近の履歴を整形 (例: 2往復分)
    recent_history_entries = chat_history[-4:]
    recent_history_text = cache_manager._format_history_for_prompt(recent_history_entries)

    if not recent_history_text.strip():
        print("追跡質問ボタン生成スキップ: 履歴テキスト空。")
        return

    # プロンプト生成とLLM呼び出し
    button_prompt = config.FOLLOW_UP_PROMPT.format(
        max_buttons=config.MAX_FOLLOW_UP_BUTTONS,
        recent_history_text=recent_history_text
    )
    follow_up_suggestions_raw = await llm_manager.generate_lowload_response(button_prompt)
    follow_up_suggestions = str(follow_up_suggestions_raw) if follow_up_suggestions_raw else ""

    if follow_up_suggestions and "提案なし" not in follow_up_suggestions.lower() and not llm_manager.is_error_message(follow_up_suggestions):
        follow_up_prompts = [line.strip() for line in follow_up_suggestions.split('\n') if line.strip()][:config.MAX_FOLLOW_UP_BUTTONS]
        follow_up_prompts = [p for p in follow_up_prompts if len(p) >= 3] # 短すぎる候補を除外

        if follow_up_prompts:
             print(f"生成された追跡質問候補: {follow_up_prompts}")
             try:
                 # WebhookMessageの場合でもeditできることを期待
                 view = FollowUpView(original_message=message_to_edit, follow_up_prompts=follow_up_prompts)
                 await message_to_edit.edit(view=view)
                 print("追跡質問ボタンをメッセージに追加しました。")
             except (discord.NotFound, discord.Forbidden):
                 print(f"警告: ボタン追加対象メッセージ({message_to_edit.id})が見つからないか編集権限がありません。")
             except AttributeError:
                  print(f"警告: メッセージオブジェクト ({type(message_to_edit)}) が edit メソッドをサポートしていません。")
             except Exception as e:
                 print(f"エラー: 追跡質問ボタンのメッセージへの追加中にエラー: {e}")
        else:
            print("低負荷モデルから有効な追跡質問候補が得られませんでした。")
    # else: # 提案なし or エラー (ログ抑制)