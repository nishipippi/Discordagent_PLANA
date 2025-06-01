import discord
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from datetime import datetime, timedelta, timezone
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools.brave_search import brave_search # 作成した検索ツールをインポート
from tools.memory_tools import remember_information_func, recall_information_func, RememberInput, RecallInput # 記憶・想起ツール関数とInputスキーマをインポート
from tools.general_chat import general_chat_func, GeneralChatInput, general_chat_tool # 雑談ツール関数とInputスキーマ、ツールオブジェクトをインポート
from pathlib import Path # ファイルパス操作のため
from tools.db_utils import initialize_db # データベース初期化関数をインポート
from functools import partial # LLMをツールにバインドするために使用
from langchain_core.tools import Tool, StructuredTool # ToolクラスとStructuredToolをインポート

# 0. 環境変数の読み込み
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path, override=True)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_PRIMARY_MODEL") # または GEMINI_LOWLOAD_MODEL

# --- ファイルパス設定 ---
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
SYSTEM_INSTRUCTION_PATH = PROMPTS_DIR / "system_instruction.txt"

# --- グローバル変数 ---
llm = None # グローバルでLLMオブジェクトを保持
channel_memories = {} # チャンネルごとの会話履歴を保持
SYSTEM_INSTRUCTIONS_TEMPLATE = "" # システム指示のテンプレートを保持する変数

# --- プロンプト読み込みユーティリティ ---
def load_prompt_file(file_path: Path) -> str:
    """指定されたパスのプロンプトファイルを読み込む"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return "" # ファイルが見つからない場合は空文字列を返す
    except Exception as e:
        print(f"Error loading prompt from {file_path}: {e}")
        return "" # エラー発生時も空文字列を返す

async def load_system_instructions_template():
    global SYSTEM_INSTRUCTIONS_TEMPLATE
    SYSTEM_INSTRUCTIONS_TEMPLATE = load_prompt_file(SYSTEM_INSTRUCTION_PATH)
    if not SYSTEM_INSTRUCTIONS_TEMPLATE:
        print(f"Warning: System instructions template not loaded from {SYSTEM_INSTRUCTION_PATH}. Using default.")
        SYSTEM_INSTRUCTIONS_TEMPLATE = "あなたは親切で少しユーモラスなDiscord AIアシスタントです。ユーザーの質問に丁寧に答えます。必要に応じてツールを使用してください。"
    else:
        print(f"Successfully loaded system instructions template from {SYSTEM_INSTRUCTION_PATH}")

# --- LangChain & Gemini 設定 ---
async def initialize_llm():
    global llm
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set.")
        return False
    if not GEMINI_MODEL:
        print("Error: GEMINI_PRIMARY_MODEL (or LOWLOAD_MODEL) is not set.")
        return False

    try:
        print(f"Initializing Gemini Model: {GEMINI_MODEL}")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        # 疎通確認
        response = await llm.ainvoke("こんにちは！")
        if response and hasattr(response, 'content'):
            print(f"Gemini LLM initialized successfully. Test response: {response.content[:50]}...")
            return True
        else:
            print("Gemini LLM initialization failed: No content in test response.")
            return False
    except Exception as e:
        print(f"Error initializing LangChain/Gemini LLM: {e}")
        return False

# --- Discord Bot 設定 ---
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    initialize_db() # データベースを初期化
    await load_system_instructions_template() # Bot起動時に指示テンプレートを読み込む
    if await initialize_llm():
        print("Bot is ready and LLM is initialized.")
    else:
        print("Bot is ready, but LLM initialization failed. Check API keys and model name.")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Botがメンションされているか、またはDMでメッセージが送られてきた場合
    # client.user が None でないことを確認
    if client.user and (client.user in message.mentions or isinstance(message.channel, discord.DMChannel)):
        channel_info = "DM"
        if isinstance(message.channel, discord.TextChannel) or isinstance(message.channel, discord.GroupChannel) or isinstance(message.channel, discord.VoiceChannel):
            channel_info = message.channel.name
        print(f"Mentioned by: {message.author} in {channel_info}")

        if llm is None:
            await message.channel.send("AIモデルがまだ準備できていません。しばらくお待ちください。")
            return

        # メンション部分を削除してユーザーのメッセージを取得 (DMの場合はメンションがないのでそのまま)
        content = message.content
        if client.user in message.mentions:
            content = content.replace(f'<@{client.user.id}>', '').strip()

        if not content:
            await message.channel.send(f"こんにちは、{message.author.name}さん！何かお手伝いできることはありますか？")
            return

        print(f"User message (after mention removal): '{content}'") # 追加

        try:
            # チャンネルのメモリを取得または初期化
            if message.channel.id not in channel_memories:
                memory = ConversationBufferMemory(
                    memory_key="chat_history", # MessagesPlaceholderのvariable_nameと合わせる
                    return_messages=True # メッセージオブジェクトのリストとして履歴を返す
                )
                channel_memories[message.channel.id] = memory

                # 初回のみ、Discordの過去10分間の会話履歴をロード
                # Bot自身のメッセージは除外
                # ユーザーのメッセージはHumanMessage、BotのメッセージはAIMessageとして追加
                time_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
                # limitを増やすことで、より多くの履歴を取得できるが、トークン制限に注意
                async for msg_from_history in message.channel.history(limit=50, after=time_threshold, oldest_first=True):
                    if msg_from_history.id == message.id: # 現在のメッセージは含めない
                        continue
                    if msg_from_history.author == client.user:
                        # BotのメッセージはAIMessageとして追加
                        memory.chat_memory.add_ai_message(msg_from_history.content)
                    elif not msg_from_history.author.bot: # Bot以外のユーザーメッセージ
                        # ユーザーのメッセージはHumanMessageとして追加
                        memory.chat_memory.add_user_message(msg_from_history.content)
            else:
                memory = channel_memories[message.channel.id]

            # ツールを定義
            # LLMインスタンスをツール関数にバインドし、Toolオブジェクトとしてラップ
            # ツールを定義
            # LLMインスタンスとDiscordコンテキスト情報をツール関数にバインドし、Toolオブジェクトとしてラップ
            current_server_id = str(message.guild.id) if message.guild else "DM_Channel"
            current_channel_id = str(message.channel.id)
            current_user_id = str(message.author.id)

            remember_tool = StructuredTool.from_function(
                func=partial(remember_information_func, llm=llm, server_id=current_server_id, channel_id=current_channel_id, user_id=current_user_id),
                name="remember_information",
                description=(
                    "ユーザーから提供された情報を記憶します。このツールは、メッセージが送信されたサーバーID、チャンネルID、ユーザーIDを自動的に取得します。日付、時間、場所、イベント名、参加者、詳細などの情報を記憶できます。例えば、「6/16に予算採択の打ち上げがあるから覚えといて。場所は3階ラウンジね」という指示の場合、記憶する内容の要約を`memory_key`に（例: '6/16 予算採択打ち上げ'）、具体的な詳細を`content_to_remember`に（例: '6月16日、予算採択の打ち上げ、場所は3階ラウンジ'）渡してください。授業日程表、連絡先リスト、重要なメモなどを構造化して保存するのに適しています。"
                ),
                args_schema=RememberInput
            )
            recall_tool = StructuredTool.from_function(
                func=partial(recall_information_func, llm=llm, server_id=current_server_id),
                name="recall_information",
                description=(
                    "以前に記憶した情報について質問に答えます。例えば、『月曜日の1限は何？』や『山田太郎さんの電話番号を教えて』のように使います。"
                    "何について知りたいかを具体的に質問してください。"
                ),
                args_schema=RecallInput
            )

            # general_chat_toolをLLMにバインドしてツールリストに追加
            general_chat_tool_bound = Tool(
                name=general_chat_tool.name,
                description=general_chat_tool.description,
                func=partial(general_chat_func, llm=llm),
                args_schema=general_chat_tool.args_schema
            )

            tools = [brave_search, remember_tool, recall_tool, general_chat_tool_bound]

            # Client IDをシステム指示テンプレートに埋め込む
            current_system_instructions = SYSTEM_INSTRUCTIONS_TEMPLATE.replace("{client_id}", str(client.user.id))

            # ペルソナ、会話履歴、ユーザーのメッセージ、Agentの思考過程を含むプロンプトを作成
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", current_system_instructions), # フォーマット済みの指示を使用
                MessagesPlaceholder(variable_name="chat_history"), # 会話履歴のプレースホルダー
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad") # Agentの思考過程
            ])

            # Agentの作成
            agent = create_tool_calling_agent(llm, tools, prompt_template)
            # AgentExecutorにmemoryを渡すことで、自動的に履歴が管理される
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True) # handle_parsing_errorsを追加

            # AgentExecutorに渡す情報を準備
            agent_input = {
                "input": content, # ユーザーのメッセージ
            }
            print(f"Agent input prepared: {agent_input}") # 追加

            # AgentExecutorを実行 (chat_historyはmemoryから自動的に供給されるため、ここでは渡さない)
            response = await agent_executor.ainvoke(agent_input)

            if response and "output" in response: # AgentExecutorの応答は辞書形式で'output'キーを持つ
                await message.channel.send(str(response["output"])) # 明示的に文字列に変換
                print(f"Sent Agent's reply: {str(response['output'])[:100]}...")
            else:
                await message.channel.send("AIからの応答がありませんでした。")
                print(f"AIからの応答が予期した形式ではありません。Raw response: {response}")


        except discord.errors.Forbidden:
            channel_name_for_error = "DM"
            if isinstance(message.channel, discord.TextChannel) or isinstance(message.channel, discord.GroupChannel) or isinstance(message.channel, discord.VoiceChannel):
                channel_name_for_error = message.channel.name
            print(f"Error: Missing permissions to send message in {channel_name_for_error}")
            await message.channel.send("メッセージを送信する権限がありません。")
        except Exception as e:
            print(f"Error processing message with Gemini: {e}")
            await message.channel.send(f"エラーが発生しました: {e}")

# --- Botの実行 ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN is not set. Please check your .env file.")
    else:
        try:
            client.run(DISCORD_TOKEN)
        except discord.errors.LoginFailure:
            print("Error: Failed to log in. Please check your DISCORD_TOKEN.")
        except Exception as e:
            print(f"Error running bot: {e}")
