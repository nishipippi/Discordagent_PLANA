import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import ( # 新しいノードをインポート
    fetch_chat_history,
    process_attachments_node, # 追加
    decide_tool_or_direct_response_node,
    execute_tool_node,
    generate_final_response_node,
    set_bot_instance_for_nodes # 追加
)
from langchain_core.messages import HumanMessage, AIMessage
# tools.discord_tools は nodes.py の fetch_chat_history で使用
# from tools.discord_tools import get_discord_messages
from tools.db_utils import init_db, load_chat_history, save_chat_history
from tools.timer_tools import create_timer_tool # create_timer_tool をインポート
from typing import Dict, Optional, Literal, Any, List # List を確認
import logging # logging をインポート
import base64 # base64 をインポート
import io # io をインポート
import aiohttp # aiohttp をインポート (非同期HTTPリクエスト用)

from tools.vector_store_utils import VectorStoreManager
from tools.brave_search import BraveSearchTool # BraveSearchTool クラスをインポート
from tools.memory_tools import create_memory_tools # <<< 変更: create_memory_tools をインポート
from tools.image_generation_tools import image_generation_tool
from langchain_core.tools import BaseTool

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギング設定を追加
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # logger を定義

# Discord Botトークンを取得
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# インテントを設定
intents = discord.Intents.default()
intents.message_content = True  # メッセージ内容の読み取りを許可

# Botのインスタンスを作成
class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.tool_map: Dict[str, BaseTool] = {} # Botインスタンスにツールマップを持たせる

    async def setup_hook(self):
        # データベースの初期化
        init_db()
        print("データベースの準備完了。")

        # ベクトルストアの初期化
        try:
            self.vector_store_manager = VectorStoreManager() # ここで初期化
            print("ベクトルストアの準備完了。")

            # VectorStoreManager が初期化された後にツールを準備
            if self.vector_store_manager:
                remember_tool_instance, recall_tool_instance = create_memory_tools(self.vector_store_manager)
                temp_tools = [
                    BraveSearchTool(),
                    remember_tool_instance,
                    recall_tool_instance,
                    image_generation_tool,
                    create_timer_tool(self) # self (botインスタンス) を渡す
                ]
                self.tool_map = {tool.name: tool for tool in temp_tools}
                set_bot_instance_for_nodes(self, self.tool_map) # ノードにツールマップを設定
                print("ツールマップが正常に初期化されました。")
            else:
                # VectorStoreManager がない場合のフォールバック (記憶ツールなし)
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

# ALL_TOOLS と tool_map のグローバルな定義は削除し、botインスタンスの属性として管理

# LangGraphグラフの構築
workflow = StateGraph(AgentState)

# ノードの追加
workflow.add_node("fetch_chat_history", fetch_chat_history)
workflow.add_node("process_attachments", process_attachments_node) # 新しいノードを追加
workflow.add_node("decide_action", decide_tool_or_direct_response_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("generate_response", generate_final_response_node)

# エントリーポイントの設定
workflow.set_entry_point("fetch_chat_history")

# エッジの定義
workflow.add_edge("fetch_chat_history", "process_attachments") # 履歴取得後に添付ファイル処理へ
workflow.add_edge("process_attachments", "decide_action") # 添付ファイル処理後に判断ノードへ

# decide_action ノードからの条件分岐
def select_next_node_after_decide_action(state: AgentState) -> Literal["execute_tool", "generate_response"]:
    if state.tool_name: # tool_name があればツール実行へ
        print(f"Conditional edge: Routing to execute_tool for tool: {state.tool_name}")
        return "execute_tool"
    else: # tool_name がなければ直接応答生成へ
        print("Conditional edge: Routing to generate_response (no tool called)")
        return "generate_response"

workflow.add_conditional_edges(
    "decide_action",
    select_next_node_after_decide_action,
    {
        "execute_tool": "execute_tool",
        "generate_response": "generate_response", # 直接応答の場合も generate_response へ
    }
)

# execute_tool ノードの後
workflow.add_edge("execute_tool", "generate_response") # ツール実行結果を基に応答生成

# generate_response ノードの後
workflow.add_edge("generate_response", END) # 最終応答で終了

app = workflow.compile()

@bot.event
async def on_ready():
    print(f'Botとしてログインしました: {bot.user}')
    # init_db() は setup_hook に移動したため、ここでの呼び出しは不要

    if bot.vector_store_manager:
        print("VectorStoreManager は正常に初期化されています。")
    else:
        print("警告: VectorStoreManager が初期化されていません。記憶・想起機能が動作しない可能性があります。")

@bot.event
async def on_message(message: discord.Message): # message の型ヒントを discord.Message に
    if message.author == bot.user:
        return

    if bot.user and bot.user.mentioned_in(message): # mentioned_in を使用
        user_input_text = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        channel_id = message.channel.id
        server_id = str(message.guild.id) if message.guild else "DM" # DMの場合も考慮
        user_id = str(message.author.id)
        thread_id = message.channel.id if isinstance(message.channel, discord.Thread) else None

        print(f"Received mention from {message.author.name} in channel {channel_id} (Server: {server_id})")
        print(f"User input: {user_input_text}")

        # 添付ファイルの処理
        attachments_data = []
        if message.attachments:
            print(f"Found {len(message.attachments)} attachments.")
            for attachment in message.attachments:
                try:
                    # ファイルをダウンロード
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                file_bytes = await resp.read()
                                # Base64エンコード
                                encoded_content = base64.b64encode(file_bytes).decode('utf-8')
                                
                                # 画像ファイルかPDFファイルを判別
                                if attachment.content_type and attachment.content_type.startswith('image/'):
                                    attachments_data.append({
                                        "filename": attachment.filename,
                                        "content_type": attachment.content_type,
                                        "content": encoded_content, # Base64エンコードされた画像データ
                                        "type": "image"
                                    })
                                    print(f"Processed image attachment: {attachment.filename}")
                                elif attachment.content_type == 'application/pdf':
                                    attachments_data.append({
                                        "filename": attachment.filename,
                                        "content_type": attachment.content_type,
                                        "content": encoded_content, # Base64エンコードされたPDFデータ
                                        "type": "pdf"
                                    })
                                    print(f"Processed PDF attachment: {attachment.filename}")
                                else:
                                    print(f"Skipping unsupported attachment type: {attachment.filename} ({attachment.content_type})")
                            else:
                                print(f"Failed to download attachment {attachment.filename}: Status {resp.status}")
                except Exception as e:
                    print(f"Error processing attachment {attachment.filename}: {e}")

        # 既存の会話状態をデータベースからロード
        loaded_chat_history = load_chat_history(channel_id)
        print(f"Loaded {len(loaded_chat_history)} messages from history for channel {channel_id}")

        initial_state_dict = {
            "input_text": user_input_text,
            "chat_history": loaded_chat_history + [HumanMessage(content=user_input_text)], # 現在の入力も履歴に含める
            "server_id": server_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "attachments": attachments_data, # 添付ファイルデータを追加
        }
        
        current_state = AgentState(**initial_state_dict)

        try:
            print("Invoking LangGraph app...")
            # LangGraphを実行 (チェックポイントなしのシンプルな実行)
            final_state_dict = await app.ainvoke(current_state.model_dump()) # .dict() を .model_dump() に変更
            final_state = AgentState(**final_state_dict) # AgentState に再変換
            print("LangGraph app finished.")

            # 最終的な応答を取得 (generate_final_response_node で llm_direct_response に格納される想定)
            ai_response_content = final_state.llm_direct_response
            if not ai_response_content:
                ai_response_content = "申し訳ありません、応答を生成できませんでした。"
                print("Error: llm_direct_response is empty in final_state.")
            
            print(f"Final AI response: {ai_response_content}")

            # AIの応答をチャット履歴に追加 (generate_final_response_node で既に追加されているはずなので、ここでは不要かも)
            # ただし、保存する履歴には含めたい
            history_to_save = final_state.chat_history 
            # もし generate_final_response_node で履歴に追加されていない場合はここで追加
            # if not any(isinstance(msg, AIMessage) and msg.content == ai_response_content for msg in history_to_save):
            # history_to_save.append(AIMessage(content=ai_response_content))

            save_chat_history(channel_id, history_to_save)
            print(f"Saved {len(history_to_save)} messages to history for channel {channel_id}")

            # 画像生成結果があれば画像を送信
            if final_state.image_output_base64:
                try:
                    image_bytes = base64.b64decode(final_state.image_output_base64)
                    image_file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
                    await message.channel.send(f'{message.author.mention} {ai_response_content}', file=image_file)
                    print("Generated image sent to Discord.")
                except Exception as img_e:
                    print(f"Error sending image to Discord: {img_e}")
                    await message.channel.send(f'{message.author.mention} {ai_response_content}\n(画像の送信中にエラーが発生しました。)')
            # タイマー完了メッセージの送信ロジックは tools/timer_tools.py に移動したため削除
            else:
                await message.channel.send(f'{message.author.mention} {ai_response_content}')

        except Exception as e:
            print(f"LangGraphの実行中にエラーが発生しました: {e}") # exc_info=True を削除
            await message.channel.send(f"{message.author.mention} 申し訳ありません、処理中にエラーが発生しました。")

    await bot.process_commands(message)

# Botを実行
if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("DISCORD_TOKENが設定されていません。")
