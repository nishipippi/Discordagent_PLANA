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

    # 応答をStateに保存
    return AgentState(
        user_input=user_input,
        chat_history=chat_history + [HumanMessage(content=user_input), AIMessage(content=response.content)],
        channel_id=state.channel_id,
        thread_id=state.thread_id
    )
