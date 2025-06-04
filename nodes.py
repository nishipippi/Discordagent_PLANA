import logging # logging をインポート
import json # json をインポート
from typing import Callable, List, Dict, Any, Optional, Union # Any, Optional, Union をインポート
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage # BaseMessage をインポート
from state import AgentState, ToolCall, LLMDecisionOutput # 新しく定義したPydanticモデルをインポート
from llm_config import llm_chain, llm # llm_modelをllmに変更
from tools.brave_search import BraveSearchTool # BraveSearchToolをインポート
from tools.memory_tools import remember_tool, recall_tool # 新規追加した記憶・想起ツールをインポート
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool # BaseTool クラスをインポート

logger = logging.getLogger(__name__) # logger を定義

# Brave Search Toolのインスタンス化
brave_search_tool = BraveSearchTool()

# 利用可能なツールをリストとしてまとめる (nodes.py 内で定義)
available_tools: List[BaseTool] = [brave_search_tool, remember_tool, recall_tool]
tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in available_tools}

import discord # discord.py の型ヒントのため
from discord.ext import commands # commands.Bot の型ヒントのため
from tools.discord_tools import get_discord_messages # get_discord_messages をインポート
from tools.db_utils import load_chat_history # load_chat_history をインポート

# bot インスタンスをノード内で利用するためのグローバル変数 (非推奨だが一時的に使用)
_bot_instance: Optional[commands.Bot] = None

def set_bot_instance_for_nodes(bot_instance: commands.Bot):
    global _bot_instance
    _bot_instance = bot_instance

# 新しいノード: メッセージ履歴の取得
async def fetch_chat_history(state: AgentState) -> AgentState:
    print("--- fetch_chat_history ---")
    if not _bot_instance:
        logger.error("Bot instance not set for nodes.")
        # エラーハンドリング: state にエラー情報を格納して返すなど
        # AgentState は BaseModel なので、辞書に変換して更新
        current_state_dict = state.dict() # BaseModel の .dict() メソッドを使用
        current_state_dict["chat_history"] = state.chat_history + [AIMessage(content="履歴取得エラー: Botインスタンス未設定")] # state.get() を state.chat_history に変更
        return AgentState(**current_state_dict)

    channel_id = state.channel_id # AgentState は BaseModel なので .get() ではなく直接アクセス
    # thread_id = state.thread_id # 必要に応じてスレッド対応

    # 直近のメッセージを取得 (例: 過去10件)
    new_messages = await get_discord_messages(_bot_instance, channel_id, limit=10)
    
    # 既存のチャット履歴に新しいメッセージを追加
    # 重複を避けるため、新しいメッセージが既存の履歴にないか確認するロジックを追加することも検討
    updated_chat_history = state.chat_history + new_messages
    
    # 履歴の長さを制限 (例: 最新の20件を保持)
    max_history_length = 20
    if len(updated_chat_history) > max_history_length:
        updated_chat_history = updated_chat_history[-max_history_length:]

    # AgentState の他のフィールドも維持する
    current_state_dict = state.dict() # BaseModel の .dict() メソッドを使用
    current_state_dict["chat_history"] = updated_chat_history
    return AgentState(**current_state_dict)

# decide_tool_or_direct_response_node (LLMによる判断ノード)
async def decide_tool_or_direct_response_node(state: AgentState) -> AgentState:
    print("--- decide_tool_or_direct_response_node ---")
    input_text = state.input_text
    chat_history = state.chat_history
    server_id = state.server_id
    channel_id = state.channel_id
    user_id = state.user_id

    # デバッグログ: chat_history の内容と型を確認
    logger.info(f"decide_tool_or_direct_response_node: chat_history received (length: {len(chat_history)})")
    for i, msg in enumerate(chat_history):
        logger.info(f"  [{i}] Type: {type(msg)}, Content: {msg.content[:50]}...") # content の一部を表示
        if not isinstance(msg, (HumanMessage, AIMessage, SystemMessage, BaseMessage)): # BaseMessage も含めてチェック
            logger.warning(f"  [{i}] Unexpected message type in chat_history: {type(msg)}")

    # prompts/system_instruction.txt の内容を読み込む
    try:
        with open("prompts/system_instruction.txt", "r", encoding="utf-8") as f:
            system_instruction_content = f.read()
    except FileNotFoundError:
        logger.error("prompts/system_instruction.txt not found.")
        return AgentState(
            input_text=input_text,
            chat_history=chat_history,
            server_id=server_id,
            channel_id=channel_id,
            user_id=user_id,
            llm_direct_response="エラー: システム指示プロンプトファイルが見つかりません。"
        )

    # プロンプトのプレースホルダを実際の値で置き換える
    # LLMに渡すプロンプトを構築
    formatted_system_instruction = system_instruction_content.format(
        server_id=server_id,
        channel_id=channel_id,
        user_id=user_id,
        input_text=input_text # input_text もプロンプトに渡す
    )

    messages_for_prompt: List[BaseMessage] = [SystemMessage(content=formatted_system_instruction)]
    # chat_history の各メッセージを適切な具象型に変換しつつ追加
    for msg in chat_history:
        if isinstance(msg, BaseMessage):
            # BaseMessage の type 属性を利用して具象型に変換
            if hasattr(msg, 'type'):
                if msg.type == 'human':
                    messages_for_prompt.append(HumanMessage(content=msg.content))
                elif msg.type == 'ai':
                    messages_for_prompt.append(AIMessage(content=msg.content))
                elif msg.type == 'system': # SystemMessage も考慮
                    messages_for_prompt.append(SystemMessage(content=msg.content))
                else:
                    logger.warning(f"Skipping BaseMessage with unhandled type '{msg.type}': {type(msg)}, Content: {msg.content[:50]}...")
            else:
                # type 属性がない場合は、content から推測するか、スキップ
                # ここでは安全のためスキップ
                logger.warning(f"Skipping BaseMessage without 'type' attribute: {type(msg)}, Content: {msg.content[:50]}...")
        elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)): # 既に具象型であればそのまま追加
            messages_for_prompt.append(msg)
        else:
            logger.warning(f"Skipping unexpected message type in chat_history: {type(msg)}")
            continue
    messages_for_prompt.append(HumanMessage(content=input_text)) # input_text を直接渡す

    prompt_template = ChatPromptTemplate.from_messages(messages_for_prompt)

    # LLMにツールをバインドする代わりに、structured_output を使用
    # LLMDecisionOutput がツール呼び出しの構造を定義していることを前提とする。
    # LLMはプロンプトとスキーマに基づいてツール呼び出しのJSONを生成する。
    structured_llm = llm.with_structured_output(LLMDecisionOutput)
    chain = prompt_template | structured_llm

    response_obj: Any = await chain.ainvoke({}) # LLMDecisionOutput のインスタンスとして応答を取得
    logger.info(f"LLM structured response object: {response_obj.dict()}") # 構造化された応答をログに出力

    # LLMの応答オブジェクトをパース
    try:
        thought = response_obj.thought
        tool_call_data = response_obj.tool_call
        direct_response_content = response_obj.direct_response

        print(f"LLM thought: {thought}")

        if tool_call_data: # tool_call があればツール実行へ
            tool_name = tool_call_data.name
            tool_args_raw = tool_call_data.args # 生の引数を取得

            # tool_args_raw が文字列の場合、JSONとしてパースして辞書に変換
            tool_args: Optional[Dict[str, Any]]
            if isinstance(tool_args_raw, str):
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool_args string to dict: {tool_args_raw}, Error: {e}")
                    # エラー処理: 例えば、直接応答に切り替えるか、エラーメッセージを返す
                    current_state_dict = state.dict()
                    current_state_dict["llm_direct_response"] = "AIの応答形式が予期せぬものでした。(ツール引数パースエラー)"
                    current_state_dict["tool_name"] = None
                    current_state_dict["tool_args"] = None
                    return AgentState(**current_state_dict)
            elif isinstance(tool_args_raw, dict):
                tool_args = tool_args_raw # 既に辞書型ならそのまま使用
            else:
                # 文字列でも辞書でもない予期せぬ型の場合
                logger.error(f"Unexpected type for tool_args: {type(tool_args_raw)}, value: {tool_args_raw}")
                current_state_dict = state.dict()
                current_state_dict["llm_direct_response"] = "AIの応答形式が予期せぬものでした。(ツール引数型エラー)"
                current_state_dict["tool_name"] = None
                current_state_dict["tool_args"] = None
                return AgentState(**current_state_dict)

            if tool_name and tool_args is not None: # パース後の tool_args を使用
                print(f"LLM decided to call tool: {tool_name} with args: {tool_args}")
                current_state_dict = state.dict()
                current_state_dict["tool_name"] = tool_name
                current_state_dict["tool_args"] = tool_args # パース後の tool_args をセット
                current_state_dict["llm_direct_response"] = None
                return AgentState(**current_state_dict)
            else:
                logger.warning(f"Parsed tool_call, but name or args are missing: {tool_call_data}")
                current_state_dict = state.dict()
                current_state_dict["llm_direct_response"] = "AIの応答形式が予期せぬものでした。(ツール呼び出し情報不足)"
                current_state_dict["tool_name"] = None
                current_state_dict["tool_args"] = None
                return AgentState(**current_state_dict)

        elif direct_response_content: # direct_response があれば直接応答へ
            print(f"LLM decided to respond directly: {direct_response_content}")
            current_state_dict = state.dict()
            current_state_dict["llm_direct_response"] = direct_response_content
            current_state_dict["tool_name"] = None
            current_state_dict["tool_args"] = None
            return AgentState(**current_state_dict)
        
        else: # tool_call も direct_response もない場合 (スキーマでどちらか必須にしているので、基本的には通らないはず)
            logger.error(f"LLMDecisionOutput did not contain tool_call or direct_response: {response_obj.dict()}")
            current_state_dict = state.dict()
            current_state_dict["llm_direct_response"] = "AIの応答形式が予期せぬものでした。(判断結果なし)"
            current_state_dict["tool_name"] = None
            current_state_dict["tool_args"] = None
            return AgentState(**current_state_dict)

    except Exception as e:
        logger.error(f"Error processing LLM structured response in decide_node: {e}", exc_info=True)
        current_state_dict = state.dict()
        current_state_dict["llm_direct_response"] = "AIの応答処理中にエラーが発生しました。"
        current_state_dict["tool_name"] = None
        current_state_dict["tool_args"] = None
        return AgentState(**current_state_dict)

# execute_tool_node (汎用ツール実行ノード)
async def execute_tool_node(state: AgentState) -> AgentState:
    print("--- execute_tool_node ---")
    tool_name = state.tool_name
    tool_args = state.tool_args
    input_text = state.input_text
    chat_history = state.chat_history
    server_id = state.server_id
    channel_id = state.channel_id
    user_id = state.user_id

    if not tool_name:
        logger.error("Tool name not found in state.")
        return AgentState(
            input_text=input_text,
            chat_history=chat_history,
            server_id=server_id,
            channel_id=channel_id,
            user_id=user_id,
            tool_output="エラー: 実行すべきツールが指定されていません。",
            tool_name=None,
            tool_args=None
        )

    tool = tool_map.get(tool_name)
    if not tool:
        logger.error(f"Tool '{tool_name}' not found in tool_map.")
        return AgentState(
            input_text=input_text,
            chat_history=chat_history,
            server_id=server_id,
            channel_id=channel_id,
            user_id=user_id,
            tool_output=f"エラー: ツール '{tool_name}' が見つかりません。",
            tool_name=None,
            tool_args=None
        )

    print(f"Executing tool: {tool_name} with args: {tool_args}")
    try:
        # tool_args が None でないことを確認
        if tool_args is None:
            raise ValueError("Tool arguments are None.")
        # ツールを実行 (非同期ツールを想定)
        # tool.ainvoke は辞書を引数として受け取り、args_schema に基づいて処理する
        tool_output = await tool.ainvoke(tool_args)
        print(f"Tool '{tool_name}' executed. Output: {str(tool_output)[:100]}...") # 出力の一部を表示
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        tool_output = f"エラー: ツール '{tool_name}' の実行中に問題が発生しました: {e}"

    return AgentState(
        input_text=input_text,
        chat_history=chat_history,
        server_id=server_id,
        channel_id=channel_id,
        user_id=user_id,
        tool_output=tool_output,
        tool_name=None, # ツール実行後はツール情報をクリア
        tool_args=None
    )

# generate_final_response_node (ツール結果を用いた応答生成ノード)
async def generate_final_response_node(state: AgentState) -> AgentState:
    print("--- generate_final_response_node ---")
    input_text = state.input_text
    # ★★★ chat_history をここで取得 ★★★
    current_chat_history_at_entry = list(state.chat_history) # コピーを作成して変更の影響を受けないようにする
    tool_name = state.tool_name
    tool_output = state.tool_output
    llm_direct_response = state.llm_direct_response

    logger.info("--- generate_final_response_node (ENTRY) ---")
    logger.info(f"Received state.chat_history (length: {len(current_chat_history_at_entry)}):")
    for i, item in enumerate(current_chat_history_at_entry):
        logger.info(f"  Item {i}: type={type(item)}, value='{str(item)[:100]}...'")
        if not isinstance(item, BaseMessage):
            logger.error(f"  ERROR @ ENTRY: Item {i} is NOT a BaseMessage subclass! Value: {item}")

    final_response_content = ""

    if llm_direct_response:
        final_response_content = llm_direct_response
        print(f"Direct LLM response: {final_response_content}")
    elif tool_output:
        system_message_content = (
            "あなたはDiscord AIエージェントのプラナです。ユーザーの質問に丁寧かつ的確に答えてください。"
            "以下のツール実行結果を参考に、ユーザーの質問に答えてください。結果がない場合や関連しない場合は、その旨を伝えてください。\n\n"
            f"ツール実行結果:\n{tool_output}\n\n"
            "過去の会話履歴も考慮して、自然な対話を心がけてください。"
        )
        converted_chat_history: List[BaseMessage] = []
        logger.info("--- generate_final_response_node (BEFORE CONVERSION LOOP) ---")
        logger.info(f"Processing chat_history for conversion (length: {len(current_chat_history_at_entry)}):")
        for i, msg_to_convert in enumerate(current_chat_history_at_entry):
            logger.info(f"  CONVERTING Item {i}: type={type(msg_to_convert)}, value='{str(msg_to_convert)[:100]}...'")
            if not isinstance(msg_to_convert, BaseMessage):
                logger.error(f"  ERROR @ CONVERSION: Item {i} is NOT a BaseMessage subclass! Value: {msg_to_convert}")
                continue

            if hasattr(msg_to_convert, 'type') and msg_to_convert.type == 'human':
                converted_chat_history.append(HumanMessage(content=msg_to_convert.content))
            elif hasattr(msg_to_convert, 'type') and msg_to_convert.type == 'ai':
                converted_chat_history.append(AIMessage(content=msg_to_convert.content))
            elif hasattr(msg_to_convert, 'type') and msg_to_convert.type == 'system':
                converted_chat_history.append(SystemMessage(content=msg_to_convert.content))
            elif isinstance(msg_to_convert, (HumanMessage, AIMessage, SystemMessage)):
                 converted_chat_history.append(msg_to_convert)
            else:
                logger.warning(f"Skipping unexpected/unhandled message type during conversion: {type(msg_to_convert)}")
        
        response_content_str: str = await llm_chain.ainvoke({ # 型ヒントを str に変更
            "user_input": input_text,
            "chat_history": converted_chat_history,
            "system_instruction": system_message_content
        })
        final_response_content = response_content_str # llm_chain の結果を直接使用
        print(f"LLM generated response based on tool output: {final_response_content}")
    else:
        final_response_content = "申し訳ありません、応答を生成できませんでした。"
        print("No direct response or tool output to generate final response.")

    updated_chat_history = current_chat_history_at_entry + [AIMessage(content=final_response_content)]

    return AgentState(
        input_text=input_text,
        chat_history=updated_chat_history,
        server_id=state.server_id,
        channel_id=state.channel_id,
        user_id=state.user_id,
        thread_id=state.thread_id,
        llm_direct_response=final_response_content,
        tool_name=None,
        tool_args=None,
        tool_output=None,
        search_query=None,
        search_results=None,
        should_search_decision=None
    )
