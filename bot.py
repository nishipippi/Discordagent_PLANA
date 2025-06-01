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
from pathlib import Path # ファイルパス操作のため

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

        print(f"User message: {content}")

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
            tools = [brave_search]

            # Client IDをシステム指示テンプレートに埋め込む
            current_system_instructions = SYSTEM_INSTRUCTIONS_TEMPLATE.replace("{client_id}", str(client.user.id))

            # ペルソナ、会話履歴、ユーザーのメッセージ、Agentの思考過程を含むプロンプトを作成
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", current_system_instructions), # フォーマット済みの指示を使用
                MessagesPlaceholder(variable_name="chat_history"), # 会話履歴のプレースホルダー
                HumanMessage(content=content),
                MessagesPlaceholder(variable_name="agent_scratchpad") # Agentの思考過程
            ])

            # Agentの作成
            agent = create_tool_calling_agent(llm, tools, prompt_template)
            # AgentExecutorにmemoryを渡すことで、自動的に履歴が管理される
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True) # handle_parsing_errorsを追加

            # AgentExecutorを実行 (chat_historyはmemoryから自動的に供給されるため、ここでは渡さない)
            response = await agent_executor.ainvoke({"input": content})

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
