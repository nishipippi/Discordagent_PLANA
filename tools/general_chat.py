from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

class GeneralChatInput(BaseModel):
    """ユーザーの一般的な質問や雑談に応答するための入力スキーマ。"""
    input: str = Field(description="ユーザーからの一般的な質問や雑談のメッセージ。")

def general_chat_func(input: str, llm: ChatGoogleGenerativeAI) -> str: # inputとllmを直接引数として受け取る
    """
    ユーザーの一般的な質問や雑談に応答します。
    特定の情報検索や記憶・想起を必要としない場合に利用されます。
    """
    try:
        response: BaseMessage = llm.invoke(input)
        return str(response.content)
    except Exception as e:
        return f"雑談応答の生成中にエラーが発生しました: {e}"

general_chat_tool = StructuredTool.from_function(
    func=general_chat_func,
    name="general_chat",
    description="ユーザーの一般的な質問や雑談に応答します。**他のどのツールも適用できない、または特定の情報検索や記憶・想起を必要としない場合にのみ使用してください。** 例えば、「こんにちは」や「今日の天気はどう？」のような質問に答えるのに使います。",
    args_schema=GeneralChatInput, # args_schemaを再度追加
)
