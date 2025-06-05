from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import json
import logging
import re
from functools import partial

from langchain_core.tools import BaseTool
from langchain.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from llm_config import get_google_api_key
from .db_utils import save_memory
from .vector_store_utils import VectorStoreManager
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RememberInput(BaseModel):
    """Input for the remember_information tool."""
    text_to_remember: str = Field(description="The text content that the user wants to remember.")
    server_id: str = Field(description="The ID of the Discord server (guild).")
    channel_id: str = Field(description="The ID of the Discord channel.")
    user_id: str = Field(description="The ID of the user who initiated the memory request.")

class RecallInput(BaseModel):
    """Input for the recall_information tool."""
    query: str = Field(description="The user's query or question to recall information about.")
    server_id: str = Field(description="The ID of the Discord server (guild) to scope the search.")
    user_id: str = Field(description="The ID of the user to scope the search for their memories.")

async def remember_information_func(
    text_to_remember: str,
    server_id: str,
    channel_id: str,
    user_id: str,
    vector_store_manager: VectorStoreManager, # 引数として VectorStoreManager を受け取る
) -> str:
    """
    Structures user's text using an LLM, saves it to SQLite, and embeds it in a vector store.
    """
    try:
        google_api_key = get_google_api_key()
        if not google_api_key:
            logger.error("Google API Key not found.")
            return "エラー: Google APIキーが設定されていません。"

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=google_api_key)

        try:
            with open("prompts/structure_memory_prompt.txt", "r", encoding="utf-8") as f:
                prompt_content = f.read()
        except FileNotFoundError:
            logger.error("prompts/structure_memory_prompt.txt not found.")
            return "エラー: 記憶構造化プロンプトファイルが見つかりません。"

        prompt_template = ChatPromptTemplate.from_template(prompt_content)
        chain = prompt_template | llm

        logger.info(f"Structuring memory for: {text_to_remember}")
        structured_response = await chain.ainvoke({"user_input": text_to_remember})
        structured_data_str = str(structured_response.content)

        processed_str = ""
        llm_output_str = structured_data_str.strip()

        try:
            match = re.search(r"```json\s*(.*?)\s*```", llm_output_str, re.DOTALL)
            if match:
                processed_str = match.group(1).strip()
            else:
                processed_str = llm_output_str.strip()
            structured_data_json = json.loads(processed_str)
            logger.info(f"Structured data: {structured_data_json}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: '{llm_output_str}' (processed: '{processed_str}') - Error: {e}")
            structured_data_json = {"raw_structured_text": llm_output_str, "error": "JSON parsing failed"}

        memory_id = save_memory(
            user_id=user_id,
            server_id=server_id,
            channel_id=channel_id,
            original_text=text_to_remember,
            structured_data=json.dumps(structured_data_json, ensure_ascii=False)
        )
        if memory_id is None:
            logger.info(f"Memory (key derived from '{text_to_remember[:30]}...') likely already exists or DB error for user {user_id}. Skipping vector store addition as well.")
            return "その情報は既に記憶されているか、データベースへの保存に問題がありました。"
        
        # memory_id が None でない場合のみベクトルストアに追加
        metadata = {
            "server_id": server_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "memory_db_id": memory_id,
        }
        document = Document(
            page_content=text_to_remember,
            metadata=metadata
        )
        logger.info(f"Adding document to vector store. Page content: '{document.page_content}', Metadata: {document.metadata}")
        vector_store_manager.add_documents([document])

        logger.info(f"Memory embedded and saved to vector store with ID: {memory_id}")
        return f"情報を記憶しました。(ID: {memory_id})"

    except Exception as e:
        logger.error(f"Error in remember_information_func: {e}", exc_info=True)
        return f"記憶処理中にエラーが発生しました: {e}"

async def recall_information_func(
    query: str,
    server_id: str,
    user_id: str,
    vector_store_manager: VectorStoreManager, # 引数として VectorStoreManager を受け取る
) -> str:
    """
    Recalls information from vector store and SQLite based on user query,
    then generates an answer using an LLM.
    """
    logger.info(f"Recalling information for query: '{query}' for user {user_id} in server {server_id}")
    retrieved_info_parts = []

    try:
        similar_docs_with_scores = vector_store_manager.search_similar_documents(query, k=5)

        if similar_docs_with_scores:
            logger.info(f"Found {len(similar_docs_with_scores)} similar docs from vector store for query '{query}'.")
            count = 0
            for doc, score in similar_docs_with_scores:
                logger.info(f"  Checking doc from vector store: Content='{doc.page_content[:50]}...', Score={score:.4f}, Metadata={doc.metadata}")
                logger.info(f"  Comparing with: query_user_id='{user_id}', query_server_id='{server_id}'")
                if doc.metadata.get("user_id") == user_id and doc.metadata.get("server_id") == server_id:
                    retrieved_info_parts.append(
                        f"- (類似度: {score:.4f}) 記憶された内容: {doc.page_content}\n  (DB ID: {doc.metadata.get('memory_db_id')}, チャンネル: {doc.metadata.get('channel_id')})"
                    )
                    count += 1
                    logger.info(f"  MATCH! Appended to retrieved_info_parts. Current count: {count}, retrieved_info_parts length: {len(retrieved_info_parts)}") # 追加ログ
                    if count >= 3:
                        logger.info("  Reached max relevant docs (3). Breaking loop.") # 追加ログ
                        break
                else:
                    logger.debug(f"Skipping doc due to metadata mismatch: user_id={doc.metadata.get('user_id')} vs {user_id}, server_id={doc.metadata.get('server_id')} vs {server_id}")
            
            logger.info(f"After loop - Final retrieved_info_parts length: {len(retrieved_info_parts)}") # 追加ログ
            if retrieved_info_parts:
                logger.info(f"Filtered {len(retrieved_info_parts)} relevant docs for user {user_id}, server {server_id}.")
        else:
            logger.info(f"No similar docs found in vector store for query '{query}'.")
    except Exception as e:
        logger.error(f"Error during vector store search in recall_information_func: {e}", exc_info=True)
        retrieved_info_parts.append("ベクトルストアからの情報検索中にエラーが発生しました。")

    if not retrieved_info_parts:
        logger.info(f"No relevant information found in memories for query '{query}', user {user_id}, server {server_id}.")
        return "関連する情報は見つかりませんでした。"

    try:
        google_api_key = get_google_api_key()
        if not google_api_key:
            return "エラー: Google APIキーが設定されていません。"

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=google_api_key)
        
        try:
            with open("prompts/answer_from_memory_prompt.txt", "r", encoding="utf-8") as f:
                prompt_content = f.read()
        except FileNotFoundError:
            logger.error("prompts/answer_from_memory_prompt.txt not found.")
            return "エラー: 回答生成プロンプトファイルが見つかりません。"

        prompt_template = ChatPromptTemplate.from_template(prompt_content)
        context_for_llm = "\n".join(retrieved_info_parts)
        logger.debug(f"Context for LLM (recall): {context_for_llm}")

        chain = prompt_template | llm
        response = await chain.ainvoke({
            "retrieved_memories": context_for_llm,
            "user_query": query
        })
        final_answer = str(response.content)
        logger.info(f"LLM generated answer from recall: {final_answer}")
        return final_answer

    except Exception as e:
        logger.error(f"Error during LLM answer generation in recall_information_func: {e}", exc_info=True)
        return f"回答生成中にエラーが発生しました: {e}"

def create_memory_tools(vector_store_manager_instance: VectorStoreManager) -> Tuple[BaseTool, BaseTool]:
    async def remember_wrapper(text_to_remember: str, server_id: str, channel_id: str, user_id: str):
        return await remember_information_func(
            text_to_remember=text_to_remember,
            server_id=server_id,
            channel_id=channel_id,
            user_id=user_id,
            vector_store_manager=vector_store_manager_instance
        )

    remember_tool_obj = StructuredTool.from_function(
        name="remember_information",
        description="ユーザーが指定したテキスト情報を構造化し、データベースとベクトルストアに記憶します。後で思い出せるように情報を保存したい場合に使用します。",
        args_schema=RememberInput,
        coroutine=remember_wrapper,
        handle_tool_error=True
    )

    async def recall_wrapper(query: str, server_id: str, user_id: str):
        return await recall_information_func(
            query=query,
            server_id=server_id,
            user_id=user_id,
            vector_store_manager=vector_store_manager_instance
        )

    recall_tool_obj = StructuredTool.from_function(
        name="recall_information",
        description="以前に記憶した情報に基づいてユーザーの質問に答えたり、関連情報を提供したりします。「〇〇について教えて」「前に話した△△は何だっけ？」のような場合に使用します。",
        args_schema=RecallInput,
        coroutine=recall_wrapper,
        handle_tool_error=True
    )
    return remember_tool_obj, recall_tool_obj
