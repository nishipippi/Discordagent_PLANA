import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

def load_system_instruction(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# LLMの初期化
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.0-flash"),
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# プロンプトテンプレートの作成
system_instruction = load_system_instruction("prompts/system_instruction.txt")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# LLMチェーンの作成
llm_chain = prompt | llm
