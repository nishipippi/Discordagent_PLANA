from typing import Callable, List, Dict, Any, Optional, Union
import discord
from discord.ext import commands
import logging
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from state import AgentState, ToolCall, LLMDecisionOutput
from llm_config import llm_chain, llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import BaseTool

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from state import AgentState, ToolCall, LLMDecisionOutput
from llm_config import llm_chain, llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_bot_instance: Optional[commands.Bot] = None
_tool_map: Optional[Dict[str, BaseTool]] = None

def set_bot_instance_for_nodes(bot_instance: commands.Bot, tool_map: Dict[str, BaseTool]):
    global _bot_instance
    global _tool_map
    _bot_instance = bot_instance
    _tool_map = tool_map

# 進捗メッセージ更新ヘルパー関数
async def _update_progress_message(state: AgentState, new_content: str):
    if not _bot_instance:
        logger.warning("Progress update skipped: Bot instance not set.")
        return
    if state.progress_message_id and state.progress_channel_id:
        channel = _bot_instance.get_channel(state.progress_channel_id)
        if channel and isinstance(channel, (discord.TextChannel, discord.Thread, discord.DMChannel, discord.GroupChannel)):
            try:
                progress_message = await channel.fetch_message(state.progress_message_id)
                await progress_message.edit(content=new_content)
                logger.info(f"Progress message updated: {new_content}")
            except discord.NotFound:
                logger.warning(f"Progress message (ID: {state.progress_message_id}) not found for update. Clearing from state.")
                # メッセージが見つからない場合、stateからIDをクリア
                state.progress_message_id = None 
                state.progress_channel_id = None
            except discord.Forbidden:
                logger.warning(f"Forbidden to edit progress message (ID: {state.progress_message_id}).")
            except Exception as e:
                logger.error(f"Error updating progress message (ID: {state.progress_message_id}): {e}", exc_info=True)
        else:
            logger.warning(f"Progress update skipped: Channel (ID: {state.progress_channel_id}) not found or not a messageable channel.")
    else:
        logger.info("Progress update skipped: No progress message ID or channel ID in state.")

from tools.discord_tools import get_discord_messages
from tools.db_utils import load_chat_history

async def process_attachments_node(state: AgentState) -> AgentState:
    await _update_progress_message(state, f"<@{state.user_id}> さんのために添付ファイルを処理中です...")
    print("--- process_attachments_node ---")
    attachments = state.attachments
    input_text = state.input_text
    chat_history = list(state.chat_history)

    if not attachments:
        print("No attachments found. Skipping attachment processing.")
        return state

    content_parts: List[Union[str, Dict[str, Any]]] = []
    if input_text:
        content_parts.append({"type": "text", "text": input_text})

    for attachment in attachments:
        file_type = attachment.get("type")
        filename = attachment.get("filename", "unknown_file")
        content_type = attachment.get("content_type", "application/octet-stream")
        encoded_content = attachment.get("content")

        if file_type == "image" and encoded_content:
            image_url = f"data:{content_type};base64,{encoded_content}"
            content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
            print(f"Added image attachment to content: {filename}")
        elif file_type == "pdf" and encoded_content:
            content_parts.append({
                "type": "media",
                "mime_type": "application/pdf",
                "data": encoded_content,
            })
            print(f"Added PDF attachment (Base64) to content_parts for LLM: {filename}")
        else:
            print(f"Skipping unsupported attachment type in node: {filename} ({content_type})")

    if chat_history and isinstance(chat_history[-1], HumanMessage) and chat_history[-1].content == input_text:
        if isinstance(chat_history[-1].content, str):
            print(f"Popping last HumanMessage with simple text content: {input_text}")
            chat_history.pop()

    if content_parts:
        multimodal_human_message = HumanMessage(content=content_parts)
        updated_chat_history = chat_history + [multimodal_human_message]
        print("Created and added new multimodal HumanMessage to chat_history.")
    else:
        updated_chat_history = chat_history
        print("No content_parts to create a new HumanMessage. Using existing chat_history.")

    current_state_dict = state.model_dump()
    current_state_dict["chat_history"] = updated_chat_history
    current_state_dict["attachments"] = []
    return AgentState(**current_state_dict)

async def fetch_chat_history(state: AgentState) -> AgentState:
    await _update_progress_message(state, f"<@{state.user_id}> さんのために過去の会話を読み込んでいます...")
    print("--- fetch_chat_history ---")
    if not _bot_instance:
        logger.error("Bot instance not set for nodes.")
        current_state_dict = state.dict()
        current_state_dict["chat_history"] = state.chat_history + [AIMessage(content="履歴取得エラー: Botインスタンス未設定")]
        return AgentState(**current_state_dict)

    channel_id = state.channel_id
    new_messages = await get_discord_messages(_bot_instance, channel_id, limit=10)
    
    prefixed_new_messages: List[BaseMessage] = []
    for msg in new_messages:
        if isinstance(msg, HumanMessage):
            content_str = str(msg.content) if isinstance(msg.content, list) else msg.content
            prefixed_new_messages.append(HumanMessage(content=f"[過去の会話] Human: {content_str}"))
        elif isinstance(msg, AIMessage):
            prefixed_new_messages.append(AIMessage(content=f"[過去の会話] AI: {msg.content}"))
        else:
            prefixed_new_messages.append(msg)

    updated_chat_history = state.chat_history + prefixed_new_messages
    
    max_history_length = 5
    if len(updated_chat_history) > max_history_length:
        updated_chat_history = updated_chat_history[-max_history_length:]

    current_state_dict = state.dict()
    current_state_dict["chat_history"] = updated_chat_history
    return AgentState(**current_state_dict)

async def decide_tool_or_direct_response_node(state: AgentState) -> AgentState:
    await _update_progress_message(state, f"<@{state.user_id}> さんのために次に何をすべきか考えています...")
    print("--- decide_tool_or_direct_response_node ---")
    input_text = state.input_text
    chat_history = state.chat_history
    server_id = state.server_id
    channel_id = state.channel_id
    user_id = state.user_id

    logger.info(f"decide_tool_or_direct_response_node: chat_history received (length: {len(chat_history)})")
    for i, msg in enumerate(chat_history):
        logger.info(f"  [{i}] Type: {type(msg)}, Content: {msg.content[:50]}...")
        if not isinstance(msg, (HumanMessage, AIMessage, SystemMessage, BaseMessage)):
            logger.warning(f"  [{i}] Unexpected message type in chat_history: {type(msg)}")

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

    formatted_system_instruction = system_instruction_content.format(
        server_id=server_id,
        channel_id=channel_id,
        user_id=user_id,
        input_text=input_text
    )

    messages_for_prompt: List[BaseMessage] = [SystemMessage(content=formatted_system_instruction)]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                messages_for_prompt.append(HumanMessage(content=msg.content))
            elif isinstance(msg.content, str):
                messages_for_prompt.append(HumanMessage(content=msg.content))
            else:
                logger.warning(f"HumanMessage with unexpected content type: {type(msg.content)}. Content: {str(msg.content)[:100]}...")
                messages_for_prompt.append(HumanMessage(content="[形式不明のメッセージ]"))

        elif isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                messages_for_prompt.append(AIMessage(content=msg.content))
            else:
                logger.warning(f"AIMessage with unexpected content type: {type(msg.content)}. Content: {str(msg.content)[:100]}...")
                messages_for_prompt.append(AIMessage(content="[形式不明のAI応答]"))

        elif isinstance(msg, SystemMessage):
            if isinstance(msg.content, str):
                messages_for_prompt.append(SystemMessage(content=msg.content))
            else:
                logger.warning(f"SystemMessage with unexpected content type: {type(msg.content)}. Content: {str(msg.content)[:100]}...")
                messages_for_prompt.append(SystemMessage(content="[形式不明のシステムメッセージ]"))
        
        else:
            logger.warning(f"Skipping unexpected message type in chat_history for LLM prompt: {type(msg)}")
            continue

    prompt_template = ChatPromptTemplate.from_messages(messages_for_prompt)

    structured_llm = llm.with_structured_output(LLMDecisionOutput)
    chain = prompt_template | structured_llm

    response_obj: Any = await chain.ainvoke({})
    logger.info(f"LLM structured response object: {response_obj.dict()}")

    current_state_dict = state.model_dump() # 先にダンプしておく

    try:
        thought = response_obj.thought
        tool_call_data = response_obj.tool_call
        direct_response_content = response_obj.direct_response

        print(f"LLM thought: {thought}")

        if tool_call_data:
            tool_name = tool_call_data.name
            tool_args = tool_call_data.args # この時点で tool_args は辞書のはず

            print(f"LLM decided to call tool: {tool_name} with args: {tool_args}")
            current_state_dict["tool_name"] = tool_name
            current_state_dict["tool_args"] = tool_args
            current_state_dict["llm_direct_response"] = None
            return AgentState(**current_state_dict)

        elif direct_response_content:
            print(f"LLM decided to respond directly: {direct_response_content}")
            current_state_dict["llm_direct_response"] = direct_response_content
            current_state_dict["tool_name"] = None
            current_state_dict["tool_args"] = None
            return AgentState(**current_state_dict)
        
        else:
            logger.error(f"LLMDecisionOutput did not contain tool_call or direct_response: {response_obj.dict()}")
            current_state_dict["llm_direct_response"] = "AIの応答形式が予期せぬものでした。(判断結果なし)"
            current_state_dict["tool_name"] = None
            current_state_dict["tool_args"] = None
            return AgentState(**current_state_dict)

    except Exception as e: # ここで Pydantic の ValidationError も捕捉される
        logger.error(f"Error processing LLM structured response in decide_node: {e}", exc_info=True)
        current_state_dict = state.model_dump() # state を再ダンプ
        current_state_dict["llm_direct_response"] = (
            "AIの応答を解析中に問題が発生しました。ツールを正しく使用できない可能性があります。"
            "別の方法で回答を試みます。"
        )
        current_state_dict["tool_name"] = None
        current_state_dict["tool_args"] = None
        return AgentState(**current_state_dict)

async def execute_tool_node(state: AgentState) -> AgentState:
    tool_name = state.tool_name if state.tool_name else "不明なツール"
    await _update_progress_message(state, f"<@{state.user_id}> さんのためにツール「{tool_name}」を実行中です...")
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

    if not _tool_map:
        logger.error("Tool map not set for nodes.")
        return AgentState(
            input_text=input_text,
            chat_history=chat_history,
            server_id=server_id,
            channel_id=channel_id,
            user_id=user_id,
            tool_output="エラー: ツールマップが設定されていません。",
            tool_name=None,
            tool_args=None
        )

    tool = _tool_map.get(tool_name)
    if not tool:
        logger.error(f"Tool '{tool_name}' not found in _tool_map.")
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
        if tool_args is None:
            raise ValueError("Tool arguments are None.")
        tool_output_result = await tool.ainvoke(tool_args)
        print(f"Tool '{tool_name}' executed. Output: {str(tool_output_result)[:100]}...")
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        tool_output_result = f"エラー: ツール '{tool_name}' の実行中に問題が発生しました: {e}"

    current_state_dict = state.model_dump()

    if tool_name == "web_search" and isinstance(tool_output_result, list):
        current_state_dict["search_results"] = tool_output_result
        if tool_output_result:
            first_result = tool_output_result[0]
            current_state_dict["tool_output"] = f"検索結果: {first_result.get('title', '')} - {first_result.get('url', '')}"
        else:
            current_state_dict["tool_output"] = "検索結果はありませんでした。"
    else:
        current_state_dict["tool_output"] = str(tool_output_result)

    current_state_dict["tool_name"] = None
    current_state_dict["tool_args"] = None
    
    return AgentState(**current_state_dict)

async def generate_final_response_node(state: AgentState) -> AgentState:
    await _update_progress_message(state, f"<@{state.user_id}> さんのために応答を生成中です...")
    print("--- generate_final_response_node ---")
    input_text = state.input_text
    current_chat_history_at_entry = list(state.chat_history)
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
    image_output_base64: Optional[str] = None

    if llm_direct_response:
        final_response_content = llm_direct_response
        print(f"Direct LLM response: {final_response_content}")
    elif tool_output:
        image_data_prefix = "image_base64_data::"

        if tool_output.startswith("エラー:"):
            print(f"Tool execution resulted in an error. Generating response with LLM based on: {tool_output}")
            
            system_message_content = (
                "あなたはDiscord AIエージェントのプラナです。ユーザーの質問に丁寧かつ的確に答えてください。"
                "以下のツール実行結果（エラーメッセージ）を参考に、ユーザーに状況を伝えてください。\n\n"
                f"ツール実行結果:\n{tool_output}\n\n"
                "過去の会話履歴も考慮して、自然な対話を心がけてください。"
            )
            
            converted_chat_history: List[BaseMessage] = []
            logger.info("--- generate_final_response_node (BEFORE CONVERSION LOOP for LLM call) ---")
            logger.info(f"Processing chat_history for conversion (length: {len(current_chat_history_at_entry)}):")
            for i, msg_to_convert in enumerate(current_chat_history_at_entry):
                logger.info(f"  CONVERTING Item {i}: type={type(msg_to_convert)}, value='{str(msg_to_convert.content)[:100]}...'")
                if not isinstance(msg_to_convert, BaseMessage):
                    logger.error(f"  ERROR @ CONVERSION: Item {i} is NOT a BaseMessage subclass! Value: {msg_to_convert}")
                    continue

                if isinstance(msg_to_convert, HumanMessage):
                    if isinstance(msg_to_convert.content, list):
                        text_parts = [part["text"] for part in msg_to_convert.content if isinstance(part, dict) and part.get("type") == "text"]
                        processed_content = "\n".join(text_parts)
                        has_non_text_attachment = any(
                            isinstance(part, dict) and part.get("type") != "text" for part in msg_to_convert.content
                        )
                        if has_non_text_attachment:
                            if processed_content:
                                processed_content += " [添付ファイルあり]"
                            else:
                                processed_content = "[添付ファイルあり]"
                        if not processed_content:
                            processed_content = "[内容のない添付メッセージ]"
                        converted_chat_history.append(HumanMessage(content=processed_content))
                    elif isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(HumanMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"HumanMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(HumanMessage(content="[形式不明のメッセージ]"))
                
                elif isinstance(msg_to_convert, AIMessage):
                    if isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(AIMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"AIMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(AIMessage(content="[形式不明のAI応答]"))

                elif isinstance(msg_to_convert, SystemMessage):
                    if isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(SystemMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"SystemMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(SystemMessage(content="[形式不明のシステムメッセージ]"))
                
                else:
                    logger.warning(f"Skipping unexpected/unhandled message type during conversion in generate_final_response_node: {type(msg_to_convert)}")
            
            response_content_str: str = await llm_chain.ainvoke({
                "user_input": input_text,
                "chat_history": converted_chat_history,
                "system_instruction": system_message_content
            })
            final_response_content = response_content_str
            image_output_base64 = None


        elif tool_output.startswith("タイマーを") and "に設定しました。時間になったらお知らせします。" in tool_output:
            print("Timer setup confirmation received. Setting response content.")
            final_response_content = tool_output
            image_output_base64 = None
        elif tool_output.startswith("タイマーを") and "に設定しました。時間になったらお知らせします。" in tool_output:
            print("Timer setup confirmation received. Setting response content.")
            final_response_content = tool_output
            image_output_base64 = None
        elif "Timer for" in tool_output and "has finished!" in tool_output:
            print("Timer completion notification received. Setting empty response for bot.py to handle.")
            final_response_content = ""
            image_output_base64 = None


        elif tool_output.startswith(image_data_prefix):
            print("Image generation tool output received. Setting fixed response and image data.")
            final_response_content = "画像を生成しました！"
            image_output_base64 = tool_output.split(image_data_prefix, 1)[1]
        
        else:
            print(f"Tool execution resulted in non-error, non-timer, non-image output. Generating response with LLM based on: {tool_output}")
            
            system_message_content = (
                "あなたはDiscord AIエージェントのプラナです。ユーザーの質問に丁寧かつ的確に答えてください。"
                "以下のツール実行結果を参考に、ユーザーの質問に答えてください。\n\n"
                f"ツール実行結果:\n{tool_output}\n\n"
                "過去の会話履歴も考慮して、自然な対話を心がけてください。"
            )
            
            converted_chat_history: List[BaseMessage] = []
            logger.info("--- generate_final_response_node (BEFORE CONVERSION LOOP for LLM call) ---")
            logger.info(f"Processing chat_history for conversion (length: {len(current_chat_history_at_entry)}):")
            for i, msg_to_convert in enumerate(current_chat_history_at_entry):
                logger.info(f"  CONVERTING Item {i}: type={type(msg_to_convert)}, value='{str(msg_to_convert.content)[:100]}...'")
                if not isinstance(msg_to_convert, BaseMessage):
                    logger.error(f"  ERROR @ CONVERSION: Item {i} is NOT a BaseMessage subclass! Value: {msg_to_convert}")
                    continue

                if isinstance(msg_to_convert, HumanMessage):
                    if isinstance(msg_to_convert.content, list):
                        text_parts = [part["text"] for part in msg_to_convert.content if isinstance(part, dict) and part.get("type") == "text"]
                        processed_content = "\n".join(text_parts)
                        has_non_text_attachment = any(
                            isinstance(part, dict) and part.get("type") != "text" for part in msg_to_convert.content
                        )
                        if has_non_text_attachment:
                            if processed_content:
                                processed_content += " [添付ファイルあり]"
                            else:
                                processed_content = "[添付ファイルあり]"
                        if not processed_content:
                            processed_content = "[内容のない添付メッセージ]"
                        converted_chat_history.append(HumanMessage(content=processed_content))
                    elif isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(HumanMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"HumanMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(HumanMessage(content="[形式不明のメッセージ]"))
                
                elif isinstance(msg_to_convert, AIMessage):
                    if isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(AIMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"AIMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(AIMessage(content="[形式不明のAI応答]"))

                elif isinstance(msg_to_convert, SystemMessage):
                    if isinstance(msg_to_convert.content, str):
                        converted_chat_history.append(SystemMessage(content=msg_to_convert.content))
                    else:
                        logger.warning(f"SystemMessage with unexpected content type in generate_final_response_node: {type(msg_to_convert.content)}. Content: {str(msg_to_convert.content)[:100]}...")
                        converted_chat_history.append(SystemMessage(content="[形式不明のシステムメッセージ]"))
                
                else:
                    logger.warning(f"Skipping unexpected/unhandled message type during conversion in generate_final_response_node: {type(msg_to_convert)}")
            
            response_content_str: str = await llm_chain.ainvoke({
                "user_input": input_text,
                "chat_history": converted_chat_history,
                "system_instruction": system_message_content
            })
            final_response_content = response_content_str
            image_output_base64 = None

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
        image_output_base64=image_output_base64,
        search_query=None,
        search_results=None,
        should_search_decision=None
    )

async def generate_followup_questions_node(state: AgentState) -> AgentState:
    await _update_progress_message(state, f"<@{state.user_id}> さんのために追加の質問を考えています...")
    print("--- generate_followup_questions_node ---")
    ai_final_response = state.llm_direct_response
    chat_history = state.chat_history

    if not ai_final_response:
        print("No AI final response to generate followup questions from.")
        current_state_dict = state.model_dump()
        current_state_dict["followup_questions"] = None
        return AgentState(**current_state_dict)

    try:
        with open("prompts/generate_followup_prompt.txt", "r", encoding="utf-8") as f:
            prompt_template_str = f.read()
    except FileNotFoundError:
        logger.error("prompts/generate_followup_prompt.txt not found.")
        current_state_dict = state.model_dump()
        current_state_dict["followup_questions"] = None
        return AgentState(**current_state_dict)

    history_for_prompt_list = []
    for msg in chat_history[-5:]:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, str):
                history_for_prompt_list.append(f"Human: {msg.content}")
            elif isinstance(msg.content, list):
                 text_content = " ".join([part["text"] for part in msg.content if isinstance(part, dict) and part.get("type") == "text"])
                 history_for_prompt_list.append(f"Human: {text_content} [添付ファイルあり]")
        elif isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                history_for_prompt_list.append(f"AI: {msg.content}")
    chat_history_for_followup = "\n".join(history_for_prompt_list)

    prompt = PromptTemplate.from_template(prompt_template_str)
    
    chain = prompt | llm 
    
    try:
        response_content = await chain.ainvoke({
            "chat_history_for_followup": chat_history_for_followup,
            "ai_final_response": ai_final_response
        })
        
        if isinstance(response_content, AIMessage):
            generated_json_str = response_content.content
        elif isinstance(response_content, str):
            generated_json_str = response_content
        else:
            raise ValueError(f"Unexpected response type from LLM: {type(response_content)}")

        print(f"LLM raw response for followup questions: {generated_json_str}")
        
        if "```json" in generated_json_str:
            generated_json_str = generated_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in generated_json_str:
             generated_json_str = generated_json_str.split("```")[1].strip()


        followup_questions_list = json.loads(generated_json_str)
        if isinstance(followup_questions_list, list) and all(isinstance(q, str) for q in followup_questions_list):
            print(f"Generated followup questions: {followup_questions_list}")
            current_state_dict = state.model_dump()
            current_state_dict["followup_questions"] = followup_questions_list[:3]
            return AgentState(**current_state_dict)
        else:
            raise ValueError("LLM did not return a valid list of strings for followup questions.")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for followup questions: {generated_json_str}, Error: {e}")
    except Exception as e:
        logger.error(f"Error generating followup questions: {e}", exc_info=True)
    
    current_state_dict = state.model_dump()
    current_state_dict["followup_questions"] = None
    return AgentState(**current_state_dict)
