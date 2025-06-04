from typing import Callable, List, Dict, Any # Any をインポート
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from llm_config import llm_chain, llm # llm_modelをllmに変更
from tools.brave_search import BraveSearchTool # BraveSearchToolをインポート
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Brave Search Toolのインスタンス化
brave_search_tool = BraveSearchTool()

# should_search_node で使用するシステム指示の文字列を定数として定義
_SHOULD_SEARCH_SYSTEM_INSTRUCTION = "あなたはユーザーの質問に対してウェブ検索が必要かどうかを判断するAIアシスタントです。検索が必要な場合は'SEARCH_REQUIRED: [検索クエリ]'と、不要な場合は'NO_SEARCH_REQUIRED'とだけ答えてください。"

# should_search_node のプロンプト
should_search_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SHOULD_SEARCH_SYSTEM_INSTRUCTION), # 定数を使用
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{user_input}")
])

# should_search_node
async def should_search_node(state: AgentState) -> AgentState:
    print("--- should_search_node ---")
    user_input = state.user_input
    chat_history = state.chat_history

    # LLMに検索の必要性を判断させる
    response: Any = await llm_chain.ainvoke({ # invoke を ainvoke に変更
        "user_input": user_input,
        "chat_history": chat_history,
        "system_instruction": _SHOULD_SEARCH_SYSTEM_INSTRUCTION # 定数を直接使用
    })
    
    decision_text = response.strip() # .content を削除
    print(f"LLM Search Decision: {decision_text}")

    search_query = None
    should_search_decision = "no"

    if decision_text.startswith("SEARCH_REQUIRED:"):
        should_search_decision = "yes"
        search_query = decision_text.replace("SEARCH_REQUIRED:", "").strip()
        print(f"Search required. Query: {search_query}")
    else:
        print("No search required.")

    return AgentState(
        user_input=user_input,
        chat_history=chat_history,
        channel_id=state.channel_id,
        thread_id=state.thread_id,
        search_query=search_query,
        should_search_decision=should_search_decision
    )

# execute_search_node
async def execute_search_node(state: AgentState) -> AgentState:
    print("--- execute_search_node ---")
    search_query = state.search_query
    
    if not search_query:
        print("No search query found. Skipping search.")
        return AgentState(
            user_input=state.user_input,
            chat_history=state.chat_history,
            channel_id=state.channel_id,
            thread_id=state.thread_id,
            search_results=[{"error": "検索クエリが指定されていません。"}] # エラーを結果として格納
        )

    print(f"Executing search for: {search_query}")
    results = await brave_search_tool.arun(search_query) # 非同期実行
    print(f"Search results obtained: {len(results)} items")

    return AgentState(
        user_input=state.user_input,
        chat_history=state.chat_history,
        channel_id=state.channel_id,
        thread_id=state.thread_id,
        search_results=results
    )

# call_llm (修正版)
async def call_llm(state: AgentState) -> AgentState:
    print("--- call_llm ---")
    user_input = state.user_input
    chat_history = state.chat_history
    search_results = state.search_results

    # プロンプトに検索結果を含めるかどうかを判断
    if search_results:
        # 検索結果を整形してプロンプトに含める
        formatted_results = "\n".join([
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}\n---"
            for r in search_results[:5] # 上位5件までを考慮
        ])
        system_message_content = (
            "あなたはDiscord AIエージェントのプラナです。ユーザーの質問に丁寧かつ的確に答えてください。"
            "以下の検索結果を参考に、ユーザーの質問に答えてください。検索結果がない場合や関連しない場合は、その旨を伝えてください。\n\n"
            f"検索結果:\n{formatted_results}\n\n"
            "過去の会話履歴も考慮して、自然な対話を心がけてください。"
        )
    else:
        system_message_content = (
            "あなたはDiscord AIエージェントのプラナです。ユーザーの質問に丁寧かつ的確に答えてください。"
            "過去の会話履歴も考慮して、自然な対話を心がけてください。"
        )

    # LLMチェーンを実行
    # llm_chainを使用
    response: Any = await llm_chain.ainvoke({ # invoke を ainvoke に変更
        "user_input": user_input,
        "chat_history": chat_history,
        "system_instruction": system_message_content # 動的に生成したシステムプロンプトを渡す
    })

    # LLMの応答をチャット履歴に追加して返す
    updated_chat_history = chat_history + [AIMessage(content=response)] # responseを直接使用

    return AgentState(
        user_input=user_input,
        chat_history=updated_chat_history,
        channel_id=state.channel_id,
        thread_id=state.thread_id,
        search_query=state.search_query, # 検索関連のstateも引き継ぐ
        search_results=state.search_results,
        should_search_decision=state.should_search_decision
    )
