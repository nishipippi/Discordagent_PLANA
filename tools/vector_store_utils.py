import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr # pydantic から SecretStr をインポート
from typing import List, Tuple

from llm_config import get_google_api_key

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_store')
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index_gemini")

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "models/embedding-001"):
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        google_api_key_str = get_google_api_key()
        if not google_api_key_str:
            raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")
        # SecretStr でラップして渡す
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=SecretStr(google_api_key_str) # SecretStr でラップ
        )
        self.index_path = FAISS_INDEX_PATH
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self) -> FAISS:
        if os.path.exists(f"{self.index_path}.faiss") and os.path.exists(f"{self.index_path}.pkl"):
            print(f"既存のベクトルストアをロードします: {self.index_path}")
            try:
                return FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"ベクトルストアのロードに失敗しました: {e}。新しいストアを作成します。")
                dummy_doc = [Document(page_content="initial document for new store")]
                return FAISS.from_documents(dummy_doc, self.embeddings)
        else:
            print(f"新しいベクトルストアを作成します: {self.index_path}")
            dummy_doc = [Document(page_content="initial document")]
            return FAISS.from_documents(dummy_doc, self.embeddings)

    def add_documents(self, documents: List[Document]):
        if not documents:
            print("追加するドキュメントがありません。")
            return
        if not self.vector_store:
            print("ベクトルストアが初期化されていません。add_documents を実行できません。")
            return
        self.vector_store.add_documents(documents)
        self.save_vector_store()
        print(f"{len(documents)} 件のドキュメントをベクトルストアに追加しました。")

    def search_similar_documents(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            print("ベクトルストアが初期化されていません。検索を実行できません。")
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"「{query}」の検索結果: {len(results)} 件")
            return results
        except Exception as e:
            print(f"類似ドキュメント検索中にエラーが発生しました: {e}")
            return []

    def save_vector_store(self):
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
            print(f"ベクトルストアを保存しました: {self.index_path}")
        else:
            print("保存するベクトルストアがありません。")
