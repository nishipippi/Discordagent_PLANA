import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser # StrOutputParserをインポート

load_dotenv()

def get_google_api_key() -> str:
    """Google APIキーを環境変数から取得する"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 環境変数が設定されていません。")
    return api_key

def get_comfyui_url() -> str:
    """ComfyUI URLを環境変数から取得する"""
    comfyui_url = os.getenv("COMFYUI_URL")
    if not comfyui_url:
        raise ValueError("COMFYUI_URL 環境変数が設定されていません。")
    return comfyui_url

def load_system_instruction(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# LLMの初期化
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash-preview-05-20"), # デフォルト値を .env.template に合わせる
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
    # generation_config={"response_mime_type": "application/json"} # with_structured_output を使用するため削除
)

# プロンプトテンプレートの作成
# system_instruction は動的に渡されるように変更
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_instruction}"), # ★★★ ここを修正 ★★★
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# LLMチェーンの作成
llm_chain = prompt | llm | StrOutputParser() # StrOutputParserを追加
