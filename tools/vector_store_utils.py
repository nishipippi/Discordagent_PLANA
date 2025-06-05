import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
from typing import List, Tuple
import logging

from llm_config import get_google_api_key

logger = logging.getLogger(__name__)

# プロジェクトルートからの相対パスでデータディレクトリを指定
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'vector_store_project_data') # 新しいサブディレクトリ名
DEFAULT_FAISS_INDEX_NAME = "faiss_index_project_data" # デフォルトのインデックス名

logger.info(f"Vector Store Dir: {VECTOR_STORE_DIR}")

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "models/embedding-001", index_name: str = DEFAULT_FAISS_INDEX_NAME):
        self.vector_store_folder = os.path.abspath(VECTOR_STORE_DIR) # 保存先フォルダを絶対パスで保持
        self.index_name = index_name # インデックス名 (ファイル名のベース)
        
        try:
            os.makedirs(self.vector_store_folder, exist_ok=True)
            logger.info(f"Ensured directory exists: {self.vector_store_folder}")
        except OSError as e:
            logger.error(f"Could not create directory {self.vector_store_folder}: {e}", exc_info=True)
            raise

        google_api_key_str = get_google_api_key()
        if not google_api_key_str:
            logger.error("GOOGLE_API_KEY 環境変数が設定されていません。")
            raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=SecretStr(google_api_key_str)
        )
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self) -> FAISS:
        faiss_file = os.path.join(self.vector_store_folder, f"{self.index_name}.faiss")
        pkl_file = os.path.join(self.vector_store_folder, f"{self.index_name}.pkl")
        logger.info(f"Attempting to load vector store from folder: {self.vector_store_folder}, index_name: {self.index_name}")
        logger.info(f"Checking for files: {faiss_file}, {pkl_file}")

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            logger.info(f"Found existing vector store files. Attempting to load.")
            try:
                store = FAISS.load_local(
                    folder_path=self.vector_store_folder, 
                    embeddings=self.embeddings, 
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Successfully loaded existing vector store.")
                return store
            except Exception as e:
                logger.error(f"Failed to load existing vector store: {e}. Creating a new store.", exc_info=True)
                dummy_doc = [Document(page_content="initial document for new store after load fail", metadata={})]
                return FAISS.from_documents(dummy_doc, self.embeddings)
        else:
            logger.info(f"Vector store files not found. Creating a new store.")
            dummy_doc = [Document(page_content="initial document for new store", metadata={})]
            return FAISS.from_documents(dummy_doc, self.embeddings)

    def add_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to add to vector store.")
            return
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot add documents.")
            return
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} document(s) to in-memory vector store.")
            self.save_vector_store()
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)

    def search_similar_documents(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot perform search.")
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Search for '{query}' found {len(results)} similar documents.")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search for '{query}': {e}", exc_info=True)
            return []

    def save_vector_store(self):
        if self.vector_store:
            logger.info(f"Attempting to save vector store to folder: {self.vector_store_folder}, with index_name: {self.index_name}")
            try:
                self.vector_store.save_local(folder_path=self.vector_store_folder, index_name=self.index_name)
                logger.info(f"Successfully saved vector store.")
                
                faiss_file_exists = os.path.exists(os.path.join(self.vector_store_folder, f"{self.index_name}.faiss"))
                pkl_file_exists = os.path.exists(os.path.join(self.vector_store_folder, f"{self.index_name}.pkl"))
                logger.info(f"Post-save check: {self.index_name}.faiss exists: {faiss_file_exists}")
                logger.info(f"Post-save check: {self.index_name}.pkl exists: {pkl_file_exists}")
                if not faiss_file_exists or not pkl_file_exists:
                    logger.error("CRITICAL: Vector store files do NOT exist immediately after saving!")
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}", exc_info=True)
        else:
            logger.warning("No vector store instance to save.")
