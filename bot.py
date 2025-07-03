import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    fetch_chat_history,
    process_attachments_node,
    decide_tool_or_direct_response_node,
    execute_tool_node,
    generate_final_response_node,
    generate_followup_questions_node,
    set_bot_instance_for_nodes
)
from langchain_core.messages import HumanMessage, AIMessage
from tools.db_utils import init_db, load_chat_history, save_chat_history
from tools.timer_tools import create_timer_tool
from typing import Dict, Optional, Literal, Any, List
import logging
import base64
import io
import aiohttp

from tools.vector_store_utils import VectorStoreManager
from tools.brave_search import BraveSearchTool
from tools.memory_tools import create_memory_tools
from tools.image_generation_tools import image_generation_tool
from langchain_core.tools import BaseTool
from discord.ui import View, Button

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.tool_map: Dict[str, BaseTool] = {}

    async def setup_hook(self):
        init_db()
        print("データベースの準備完了。")

        try:
            self.vector_store_manager = VectorStoreManager()
            print("ベクトルストアの準備完了。")

            if self.vector_store_manager:
                remember_tool_instance, recall_tool_instance = create_memory_tools(self.vector_store_manager)
                temp_tools = [
                    BraveSearchTool(),
                    remember_tool_instance,
                    recall_tool_instance,
                    image_generation_tool,
                    create_timer_tool(self)
                ]
                self.tool_map = {tool.name: tool for tool in temp_tools}
                set_bot_instance_for_nodes(self, self.tool_map)
                print("ツールマップが正常に初期化されました。")
            else:
                print("警告: VectorStoreManager が初期化できなかったため、記憶・想起ツールは利用できません。")
                temp_tools = [
                    BraveSearchTool(),
                    image_generation_tool,
                    create_timer_tool(self)
                ]
                self.tool_map = {tool.name: tool for tool in temp_tools}
                set_bot_instance_for_nodes(self, self.tool_map)

        except ValueError as e:
            print(f"ベクトルストアの初期化に失敗しました: {e}")
        except Exception as e:
            logger.exception(f"予期せぬエラーでベクトルストアの初期化に失敗: {e}")

bot = MyBot(command_prefix='!', intents=intents)

workflow = StateGraph(AgentState)

workflow.add_node("fetch_chat_history", fetch_chat_history)
workflow.add_node("process_attachments", process_attachments_node)
workflow.add_node("decide_action", decide_tool_or_direct_response_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("generate_response", generate_final_response_node)
workflow.add_node("generate_followups", generate_followup_questions_node)

workflow.set_entry_point("fetch_chat_history")

workflow.add_edge("fetch_chat_history", "process_attachments")
workflow.add_edge("process_attachments", "decide_action")

def select_next_node_after_decide_action(state: AgentState) -> Literal["execute_tool", "generate_response"]:
    if state.tool_name:
        print(f"Conditional edge: Routing to execute_tool for tool: {state.tool_name}")
        return "execute_tool"
    else:
        print("Conditional edge: Routing to generate_response (no tool called)")
        return "generate_response"

workflow.add_conditional_edges(
    "decide_action",
    select_next_node_after_decide_action,
    {
        "execute_tool": "execute_tool",
        "generate_response": "generate_response",
    }
)

workflow.add_edge("execute_tool", "generate_response")
workflow.add_edge("generate_response", "generate_followups")
workflow.add_edge("generate_followups", END)

app = workflow.compile()

# Workflowファイルを動的に読み込む関数
def get_workflow_files() -> List[str]:
    workflow_dir = "workflows"
    if not os.path.exists(workflow_dir):
        return []
    return [f for f in os.listdir(workflow_dir) if f.endswith(".json")]

class WorkflowButton(Button):
    def __init__(self, label: str, custom_id: str, bot_instance: MyBot, original_message: discord.Message, positive_prompt: str, negative_prompt: str):
        super().__init__(label=label, style=discord.ButtonStyle.primary, custom_id=custom_id)
        self.bot_instance = bot_instance
        self.original_message = original_message
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

    async def callback(self, interaction: discord.Interaction):
        if self.disabled or (self.view and self.view.is_finished()):
            await interaction.response.defer()
            return

        selected_workflow_file = self.label # ボタンのラベルがファイル名
        channel_id = interaction.channel_id
        server_id = str(interaction.guild_id) if interaction.guild else "DM"
        user_id = str(interaction.user.id)
        thread_id = interaction.channel.id if isinstance(interaction.channel, discord.Thread) else None

        progress_message: Optional[discord.Message] = None
        try:
            if isinstance(interaction.channel, (discord.TextChannel, discord.DMChannel, discord.Thread)):
                progress_message = await interaction.channel.send(f"{interaction.user.mention} `{selected_workflow_file}` を使用して画像を生成中です...")
            await interaction.response.defer()
        except discord.HTTPException as e:
            print(f"進捗メッセージの送信に失敗 (WorkflowButton): {e}")
            if not interaction.response.is_done():
                 await interaction.response.send_message("処理を開始できませんでした。", ephemeral=True)
            return

        loaded_chat_history = load_chat_history(str(channel_id))
        print(f"Loaded {len(loaded_chat_history)} messages from history for channel {channel_id} (WorkflowButton)")

        # LangGraphに渡すツール呼び出し情報を構築
        tool_call_input = {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "workflow_file": selected_workflow_file
        }
        
        initial_state_dict = {
            "input_text": f"画像生成: {self.positive_prompt} (Workflow: {selected_workflow_file})", # ログ用
            "chat_history": loaded_chat_history + [HumanMessage(content=f"画像生成の指示: {self.positive_prompt}")],
            "server_id": server_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "attachments": [],
            "tool_name": "image_generation_tool", # image_generation_toolを直接呼び出す
            "tool_input": tool_call_input,
            "progress_message_id": progress_message.id if progress_message else None,
            "progress_channel_id": progress_message.channel.id if progress_message else None,
        }
        current_state = AgentState(**initial_state_dict)

        try:
            print("Invoking LangGraph app for image generation (WorkflowButton)...")
            final_state_dict = await app.ainvoke(current_state.model_dump())
            final_state = AgentState(**final_state_dict)
            print("LangGraph app finished for image generation (WorkflowButton).")

            if progress_message:
                try:
                    await progress_message.delete()
                except discord.NotFound:
                    print("進捗メッセージが見つからず削除できませんでした (WorkflowButton)。")
                except discord.Forbidden:
                    print("進捗メッセージの削除権限がありません (WorkflowButton)。")
                except Exception as e:
                    print(f"進捗メッセージの削除中にエラー (WorkflowButton): {e}")

            ai_response_content = final_state.llm_direct_response or "申し訳ありません、応答を生成できませんでした。"
            
            history_to_save = final_state.chat_history
            save_chat_history(str(channel_id), history_to_save)

            # ワークフロー選択メッセージのボタンを無効化
            if interaction.message:
                disabled_view = View(timeout=None)
                for item in interaction.message.components:
                    if isinstance(item, discord.ActionRow):
                        for child in item.children:
                            if isinstance(child, discord.ui.Button):
                                new_button = Button(label=child.label, style=discord.ButtonStyle.secondary, custom_id=child.custom_id, disabled=True)
                                disabled_view.add_item(new_button)
                try:
                    await interaction.message.edit(view=disabled_view)
                except discord.NotFound:
                    pass

            # 画像と応答を送信
            if final_state.image_output_base64:
                try:
                    image_bytes = base64.b64decode(final_state.image_output_base64)
                    image_file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
                    await interaction.followup.send(
                        f'{interaction.user.mention} {ai_response_content}',
                        file=image_file
                    )
                    print("Generated image sent to Discord (WorkflowButton).")
                except Exception as img_e:
                    print(f"Error sending image to Discord (WorkflowButton): {img_e}")
                    await interaction.followup.send(
                        f'{interaction.user.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)'
                    )
            else:
                await interaction.followup.send(
                    f'{interaction.user.mention} {ai_response_content}'
                )

        except Exception as e:
            print(f"LangGraphの実行中にエラーが発生しました (WorkflowButton): {e}")
            if progress_message:
                try:
                    await progress_message.delete()
                except Exception:
                    pass
            if not interaction.response.is_done():
                 await interaction.response.send_message(f"{interaction.user.mention} 申し訳ありません、処理中にエラーが発生しました。", ephemeral=True)
            else:
                 await interaction.followup.send(f"{interaction.user.mention} 申し訳ありません、処理中にエラーが発生しました。", ephemeral=True)


class WorkflowSelectionView(View):
    def __init__(self, original_message: discord.Message, positive_prompt: str, negative_prompt: str, bot_instance: MyBot):
        super().__init__(timeout=180)
        self.original_message = original_message
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.bot_instance = bot_instance
        self.add_workflow_buttons()

    def add_workflow_buttons(self):
        workflow_files = get_workflow_files()
        if not workflow_files:
            self.add_item(Button(label="ワークフローファイルが見つかりません", style=discord.ButtonStyle.red, disabled=True))
            return
        
        for i, wf_name in enumerate(workflow_files):
            custom_id = f"workflow_select_{self.original_message.id}_{i}"
            self.add_item(
                WorkflowButton(
                    label=wf_name,
                    custom_id=custom_id,
                    bot_instance=self.bot_instance,
                    original_message=self.original_message,
                    positive_prompt=self.positive_prompt,
                    negative_prompt=self.negative_prompt
                )
            )

    async def on_timeout(self):
        # タイムアウト時にボタンを無効化
        for item in self.children:
            if isinstance(item, Button):
                item.disabled = True
        try:
            await self.original_message.edit(content="ワークフロー選択の時間が過ぎました。", view=self)
        except discord.NotFound:
            pass # メッセージが削除されている場合は何もしない

class FollowupButton(Button):
    def __init__(self, label: str, custom_id: str, bot_instance: MyBot):
        super().__init__(label=label, style=discord.ButtonStyle.secondary, custom_id=custom_id)
        self.bot_instance = bot_instance

    async def callback(self, interaction: discord.Interaction):
        if self.disabled or (self.view and self.view.is_finished()):
            await interaction.response.defer()
            return

        user_input_text = self.label
        channel_id = interaction.channel_id
        server_id = str(interaction.guild_id) if interaction.guild else "DM"
        user_id = str(interaction.user.id)
        thread_id = interaction.channel.id if isinstance(interaction.channel, discord.Thread) else None

        progress_message: Optional[discord.Message] = None
        try:
            if isinstance(interaction.channel, (discord.TextChannel, discord.DMChannel, discord.Thread)):
                progress_message = await interaction.channel.send(f"{interaction.user.mention} `{user_input_text}` について考え中です...")
            await interaction.response.defer()
        except discord.HTTPException as e:
            print(f"進捗メッセージの送信に失敗 (FollowupButton): {e}")
            if not interaction.response.is_done():
                 await interaction.response.send_message("処理を開始できませんでした。", ephemeral=True)
            return

        loaded_chat_history = load_chat_history(str(channel_id))
        print(f"Loaded {len(loaded_chat_history)} messages from history for channel {channel_id} (Followup)")

        initial_state_dict = {
            "input_text": user_input_text,
            "chat_history": loaded_chat_history + [HumanMessage(content=str(user_input_text))],
            "server_id": server_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "attachments": [],
            "progress_message_id": progress_message.id if progress_message else None,
            "progress_channel_id": progress_message.channel.id if progress_message else None,
        }
        current_state = AgentState(**initial_state_dict)

        try:
            print("Invoking LangGraph app for followup...")
            final_state_dict = await app.ainvoke(current_state.model_dump())
            final_state = AgentState(**final_state_dict)
            print("LangGraph app finished for followup.")

            if progress_message:
                try:
                    await progress_message.delete()
                except discord.NotFound:
                    print("進捗メッセージが見つからず削除できませんでした (FollowupButton)。")
                except discord.Forbidden:
                    print("進捗メッセージの削除権限がありません (FollowupButton)。")
                except Exception as e:
                    print(f"進捗メッセージの削除中にエラー (FollowupButton): {e}")

            ai_response_content = final_state.llm_direct_response
            if not ai_response_content:
                ai_response_content = "申し訳ありません、応答を生成できませんでした。"
            
            history_to_save = final_state.chat_history
            save_chat_history(str(channel_id), history_to_save)

            followup_view_after_button_click = None
            if final_state.followup_questions:
                followup_view_after_button_click = View(timeout=180)
                for i, q_text in enumerate(final_state.followup_questions):
                    button_custom_id = f"followup_interaction_{interaction.id}_{i}"
                    followup_view_after_button_click.add_item(
                        FollowupButton(label=q_text, custom_id=button_custom_id, bot_instance=self.bot_instance)
                    )
            
            if final_state.image_output_base64:
                try:
                    image_bytes = base64.b64decode(final_state.image_output_base64)
                    image_file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
                    if followup_view_after_button_click:
                        await interaction.followup.send(
                            f'{interaction.user.mention} {ai_response_content}',
                            file=image_file,
                            view=followup_view_after_button_click
                        )
                    else:
                        await interaction.followup.send(
                            f'{interaction.user.mention} {ai_response_content}',
                            file=image_file
                        )
                    print("Generated image sent to Discord (Followup).")
                except Exception as img_e:
                    print(f"Error sending image to Discord (Followup): {img_e}")
                    if followup_view_after_button_click:
                        await interaction.followup.send(
                            f'{interaction.user.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)',
                            view=followup_view_after_button_click
                        )
                    else:
                        await interaction.followup.send(
                            f'{interaction.user.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)'
                        )
            else:
                if followup_view_after_button_click:
                    await interaction.followup.send(
                        f'{interaction.user.mention} {ai_response_content}',
                        view=followup_view_after_button_click
                    )
                else:
                    await interaction.followup.send(
                        f'{interaction.user.mention} {ai_response_content}'
                    )

            if interaction.message:
                disabled_view = View(timeout=None)
                for item in interaction.message.components:
                    if isinstance(item, discord.ActionRow):
                        for child in item.children:
                            if isinstance(child, discord.ui.Button):
                                new_button = Button(label=child.label, style=discord.ButtonStyle.secondary, custom_id=child.custom_id, disabled=True)
                                disabled_view.add_item(new_button)
                try:
                    await interaction.message.edit(view=disabled_view)
                except discord.NotFound:
                    pass

        except Exception as e:
            print(f"LangGraphの実行中にエラーが発生しました (Followup): {e}")
            if progress_message:
                try:
                    await progress_message.delete()
                except Exception:
                    pass
            if not interaction.response.is_done():
                 await interaction.response.send_message(f"{interaction.user.mention} 申し訳ありません、処理中にエラーが発生しました。", ephemeral=True)
            else:
                 await interaction.followup.send(f"{interaction.user.mention} 申し訳ありません、処理中にエラーが発生しました。", ephemeral=True)


@bot.event
async def on_ready():
    print(f'Botとしてログインしました: {bot.user}')

    if bot.vector_store_manager:
        print("VectorStoreManager は正常に初期化されています。")
    else:
        print("警告: VectorStoreManager が初期化されていません。記憶・想起機能が動作しない可能性があります。")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if bot.user and bot.user.mentioned_in(message):
        user_input_text = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        channel_id = message.channel.id
        server_id = str(message.guild.id) if message.guild else "DM"
        user_id = str(message.author.id)
        thread_id = message.channel.id if isinstance(message.channel, discord.Thread) else None

        print(f"Received mention from {message.author.name} in channel {channel_id} (Server: {server_id})")
        print(f"User input: {user_input_text}")

        progress_message: Optional[discord.Message] = None
        try:
            if isinstance(message.channel, (discord.TextChannel, discord.DMChannel, discord.Thread)):
                progress_message = await message.channel.send(f"{message.author.mention} `{user_input_text[:50]}{'...' if len(user_input_text) > 50 else ''}` について考え中です...")
        except discord.HTTPException as e:
            print(f"進捗メッセージの送信に失敗 (on_message): {e}")

        attachments_data = []
        if message.attachments:
            print(f"Found {len(message.attachments)} attachments.")
            for attachment in message.attachments:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                file_bytes = await resp.read()
                                encoded_content = base64.b64encode(file_bytes).decode('utf-8')
                                
                                if attachment.content_type and attachment.content_type.startswith('image/'):
                                    attachments_data.append({
                                        "filename": attachment.filename,
                                        "content_type": attachment.content_type,
                                        "content": encoded_content,
                                        "type": "image"
                                    })
                                    print(f"Processed image attachment: {attachment.filename}")
                                elif attachment.content_type == 'application/pdf':
                                    attachments_data.append({
                                        "filename": attachment.filename,
                                        "content_type": attachment.content_type,
                                        "content": encoded_content,
                                        "type": "pdf"
                                    })
                                    print(f"Processed PDF attachment: {attachment.filename}")
                                else:
                                    print(f"Skipping unsupported attachment type: {attachment.filename} ({attachment.content_type})")
                            else:
                                print(f"Failed to download attachment {attachment.filename}: Status {resp.status}")
                except Exception as e:
                    print(f"Error processing attachment {attachment.filename}: {e}")

        loaded_chat_history = load_chat_history(str(channel_id))
        print(f"Loaded {len(loaded_chat_history)} messages from history for channel {channel_id}")

        initial_state_dict = {
            "input_text": user_input_text,
            "chat_history": loaded_chat_history + [HumanMessage(content=user_input_text)],
            "server_id": server_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "attachments": attachments_data,
            "progress_message_id": progress_message.id if progress_message else None,
            "progress_channel_id": progress_message.channel.id if progress_message else None,
        }
        
        current_state = AgentState(**initial_state_dict)

        try:
            print("Invoking LangGraph app...")
            final_state_dict = await app.ainvoke(current_state.model_dump())
            final_state = AgentState(**final_state_dict)
            print("LangGraph app finished.")

            if progress_message:
                try:
                    await progress_message.delete()
                except discord.NotFound:
                    print("進捗メッセージが見つからず削除できませんでした (on_message)。")
                except discord.Forbidden:
                    print("進捗メッセージの削除権限がありません (on_message)。")
                except Exception as e:
                    print(f"進捗メッセージの削除中にエラー (on_message): {e}")

            ai_response_content = final_state.llm_direct_response or "申し訳ありません、応答を生成できませんでした。"
            print(f"Final AI response: {ai_response_content}")

            history_to_save = final_state.chat_history 
            save_chat_history(str(channel_id), history_to_save)
            print(f"Saved {len(history_to_save)} messages to history for channel {channel_id}")

            if final_state.tool_name == "image_generation_tool":
                # 画像生成ツールが選択された場合、ワークフロー選択ボタンを表示
                positive_prompt = ""
                negative_prompt = ""
                if final_state.tool_input:
                    positive_prompt = final_state.tool_input.get("positive_prompt", "")
                    negative_prompt = final_state.tool_input.get("negative_prompt", "")
                
                # ワークフロー選択ビューを送信
                workflow_select_view = WorkflowSelectionView(
                    original_message=message,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    bot_instance=bot
                )
                await message.channel.send(
                    f'{message.author.mention} どのワークフローで画像を生成しますか？\n\n**ポジティブプロンプト:** `{positive_prompt}`\n**ネガティブプロンプト:** `{negative_prompt}`',
                    view=workflow_select_view
                )
                print("Workflow selection view sent to Discord.")
            else:
                # 通常の応答処理
                followup_view = None
                if final_state.followup_questions:
                    followup_view = View(timeout=180)
                    for i, q_text in enumerate(final_state.followup_questions):
                        button_custom_id = f"followup_{message.id}_{i}"
                        followup_view.add_item(FollowupButton(label=q_text, custom_id=button_custom_id, bot_instance=bot))
                
                if final_state.image_output_base64:
                    try:
                        image_bytes = base64.b64decode(final_state.image_output_base64)
                        image_file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
                        if followup_view:
                            await message.channel.send(f'{message.author.mention} {ai_response_content}', file=image_file, view=followup_view)
                        else:
                            await message.channel.send(f'{message.author.mention} {ai_response_content}', file=image_file)
                        print("Generated image sent to Discord.")
                    except Exception as img_e:
                        print(f"Error sending image to Discord: {img_e}")
                        if followup_view:
                            await message.channel.send(f'{message.author.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)', view=followup_view)
                        else:
                            await message.channel.send(f'{message.author.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)')
                else:
                    if followup_view:
                        await message.channel.send(f'{message.author.mention} {ai_response_content}', view=followup_view)
                    else:
                        await message.channel.send(f'{message.author.mention} {ai_response_content}')

        except Exception as e:
            print(f"LangGraphの実行中にエラーが発生しました: {e}")
            if progress_message:
                try:
                    await progress_message.delete()
                except Exception:
                    pass
            await message.channel.send(f"{message.author.mention} 申し訳ありません、処理中にエラーが発生しました。")

    await bot.process_commands(message)

if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("DISCORD_TOKENが設定されていません。")
