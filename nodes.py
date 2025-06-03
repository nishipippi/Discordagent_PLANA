from typing import Callable
from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState
from llm_config import llm_chain

def call_llm(state: AgentState) -> AgentState:
    user_input = state.user_input
    chat_history = state.chat_history

    # LLMチェーンを実行
    response = llm_chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history
    })

    # LLMの応答をチャット履歴に追加して返す
    # bot.pyのon_messageで最終応答を追加するロジックと連携するため、
    # ここではLLMの応答を新しいAIMessageとしてchat_historyに追加し、それを返す形にする。
    updated_chat_history = chat_history + [AIMessage(content=response.content)]

    return AgentState(
        user_input=user_input,
        chat_history=updated_chat_history,
        channel_id=state.channel_id,
        thread_id=state.thread_id
    )
