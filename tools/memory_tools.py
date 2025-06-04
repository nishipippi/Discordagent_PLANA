from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import logging

from langchain_core.tools import Tool
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
) -> str:
    """
    Structures user's text using an LLM, saves it to SQLite, and embeds it in a vector store.
    """
    try:
        # 1. 情報構造化 (LLM呼び出し)
        google_api_key = get_google_api_key()
        if not google_api_key:
            logger.error("Google API Key not found.")
            return "エラー: Google APIキーが設定されていません。"

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key) # モデル名は適宜調整

        # prompts/structure_memory_prompt.txt の内容を読み込む
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
        structured_data_str = str(structured_response.content) # 明示的にstrにキャスト

        try:
            # LLMの出力がJSON形式であることを期待
            structured_data_json = json.loads(structured_data_str)
            logger.info(f"Structured data: {structured_data_json}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {structured_data_str} - Error: {e}")
            # JSONパースに失敗した場合でも、元のテキストと未加工の構造化テキストを保存することを検討
            structured_data_json = {"raw_structured_text": structured_data_str, "error": "JSON parsing failed"}


        # 2. データベース保存 (SQLite)
        memory_id = save_memory(
            user_id=user_id,
            server_id=server_id,
            channel_id=channel_id,
            original_text=text_to_remember,
            structured_data=json.dumps(structured_data_json, ensure_ascii=False)
        )
        if memory_id is None:
            logger.error("Failed to save memory to SQLite.")
            return "エラー: 記憶をデータベースに保存できませんでした。"
        logger.info(f"Memory saved to SQLite with ID: {memory_id}")


        # 3. ベクトルストア保存 (FAISS)
        vector_store_manager = VectorStoreManager()
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
        vector_store_manager.add_documents([document])

        logger.info(f"Memory embedded and saved to vector store with ID: {memory_id}")

        return f"情報を記憶しました。(ID: {memory_id})"

    except Exception as e:
        logger.error(f"Error in remember_information_func: {e}", exc_info=True)
        return f"エラーが発生しました: {e}"

remember_tool = Tool(
    name="remember_information",
    func=remember_information_func,
    description="ユーザーが指定したテキスト情報を構造化し、データベースとベクトルストアに記憶します。後で思い出せるように情報を保存したい場合に使用します。",
    args_schema=RememberInput,
)

async def recall_information_func(
    query: str,
    server_id: str,
    user_id: str,
) -> str:
    """
    Recalls information from vector store and SQLite based on user query,
    then generates an answer using an LLM.
    """
    logger.info(f"Recalling information for query: '{query}' for user {user_id} in server {server_id}")
    retrieved_info_parts = []

    # 1. ベクトルストア検索
    try:
        vector_store_manager = VectorStoreManager()
        similar_docs_with_scores = vector_store_manager.search_similar_documents(query, k=5) # 少し多めに取得

        if similar_docs_with_scores:
            logger.info(f"Found {len(similar_docs_with_scores)} similar docs from vector store.")
            count = 0
            for doc, score in similar_docs_with_scores:
                # メタデータによるフィルタリング (user_id と server_id が一致するもののみ)
                if doc.metadata.get("user_id") == user_id and doc.metadata.get("server_id") == server_id:
                    retrieved_info_parts.append(
                        f"- (類似度: {score:.4f}) 記憶された内容: {doc.page_content}\n  (DB ID: {doc.metadata.get('memory_db_id')}, チャンネル: {doc.metadata.get('channel_id')})"
                    )
                    count += 1
                    if count >= 3: # 上位3件まで採用
                        break
            if retrieved_info_parts:
                 logger.info(f"Filtered relevant docs: {len(retrieved_info_parts)}")
    except Exception as e:
        logger.error(f"Error during vector store search: {e}", exc_info=True)
        retrieved_info_parts.append("ベクトルストアからの情報検索中にエラーが発生しました。")

    if not retrieved_info_parts:
        logger.info("No relevant information found in memories.")
        return "関連する情報は見つかりませんでした。"

    # 2. 情報統合と応答生成 (LLM呼び出し)
    try:
        google_api_key = get_google_api_key()
        if not google_api_key:
            return "エラー: Google APIキーが設定されていません。"

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        
        try:
            with open("prompts/answer_from_memory_prompt.txt", "r", encoding="utf-8") as f:
                prompt_content = f.read()
        except FileNotFoundError:
            logger.error("prompts/answer_from_memory_prompt.txt not found.")
            return "エラー: 回答生成プロンプトファイルが見つかりません。"

        prompt_template = ChatPromptTemplate.from_template(prompt_content)
        
        context_for_llm = "\n".join(retrieved_info_parts)
        logger.debug(f"Context for LLM: {context_for_llm}")

        chain = prompt_template | llm
        response = await chain.ainvoke({
            "retrieved_memories": context_for_llm,
            "user_query": query
        })
        final_answer = str(response.content) # 明示的にstrにキャスト
        logger.info(f"LLM generated answer: {final_answer}")
        return final_answer

    except Exception as e:
        logger.error(f"Error during LLM answer generation: {e}", exc_info=True)
        return f"回答生成中にエラーが発生しました: {e}"

recall_tool = Tool(
    name="recall_information",
    func=recall_information_func,
    description="以前に記憶した情報に基づいてユーザーの質問に答えたり、関連情報を提供したりします。「〇〇について教えて」「前に話した△△は何だっけ？」のような場合に使用します。",
    args_schema=RecallInput,
)
