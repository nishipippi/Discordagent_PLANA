# llm_provider.py
# (APIプロバイダーのインターフェース定義)

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Literal

# bot_constants モジュールをインポートに追加
import bot_constants

# --- エラー種別定数 ---
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_INVALID_ARGUMENT = "invalid_argument"
ERROR_TYPE_BLOCKED_PROMPT = "blocked_prompt"
ERROR_TYPE_BLOCKED_RESPONSE = "blocked_response"
ERROR_TYPE_API_ERROR = "api_error"
ERROR_TYPE_INTERNAL = "internal"
ERROR_TYPE_UNKNOWN = "unknown"
ERROR_TYPE_UNSUPPORTED_FEATURE = "unsupported_feature"

class LLMProvider(ABC):
    """LLM APIプロバイダーの抽象基底クラス"""

    @abstractmethod
    async def initialize(self, api_key: str, model_config: Dict[str, str], system_prompt: str, base_url: Optional[str] = None) -> bool:
        """
        APIクライアントとモデルを初期化する
        api_key: APIキー
        model_config: {'primary': 'model_name', 'secondary': 'model_name', 'lowload': 'model_name'}
        system_prompt: システムプロンプト文字列
        base_url: APIエンドpoint URL (Optional, プロバイダーによって不要)
        戻り値: 初期化成功フラグ
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        content_parts: List[Dict[str, Any]], # 現在のユーザー入力（結合済みテキスト+添付）
        chat_history: Optional[List[Dict[str, Any]]] = None, # 過去の会話履歴（キャッシュまたはチャンネル履歴）
        deep_cache_summary: Optional[str] = None, # Deep Cacheサマリー
    ) -> Tuple[str, str]:
        """
        通常の応答を生成する。レートリミット時はフォールバックを試みる。
        content_parts: [{'text': '...'}, {'inline_data': {'mime_type': '...', 'data': b'...'}}] の形式
        chat_history: [{'role': 'user'/'model', 'parts': [...]}] の形式
        deep_cache_summary: Deep Cacheのサマリーテキスト
        戻り値: (使用したモデル名, 応答テキスト or プラナ風エラーメッセージ)
        """
        pass

    @abstractmethod
    async def generate_lowload_response(self, prompt: str) -> Optional[str]:
        """
        低負荷モデルでシンプルな応答を生成する (ボタン生成、Deep Cache用)
        戻り値: 応答テキスト or None (エラー時)
        """
        pass

    @abstractmethod
    def format_error_message(self, error_type: str, detail: Optional[str] = None) -> str:
         """
         API固有のエラー情報を、統一的なプラナ風エラーメッセージ文字列に変換する。
         detailは追加情報（例：ブロック理由、MIMEタイプなど）
         """
         pass

    @abstractmethod
    def is_rate_limit_error(self, exception: Exception) -> bool:
        """与えられた例外がレートリミットエラーか判定する"""
        pass

    @abstractmethod
    def is_invalid_argument_error(self, exception: Exception) -> bool:
        """与えられた例外が無効な引数エラーか判定する"""
        pass

    @abstractmethod
    def get_model_name(self, model_type: Literal["primary", "secondary", "lowload"]) -> Optional[str]:
        """
        指定されたタイプのモデル名を取得する。
        model_type: 'primary', 'secondary', または 'lowload'
        戻り値: モデル名 (文字列) または None (モデルが初期化されていない場合)
        """
        pass


    def _is_error_message(self, text: str) -> bool:
        """応答テキストがエラーメッセージかどうかを簡易的に判定するヘルパー"""
        error_keywords = [
            bot_constants.ERROR_MSG_MAX_TEXT_SIZE,
            bot_constants.ERROR_MSG_MAX_IMAGE_SIZE,
            bot_constants.ERROR_MSG_IMAGE_READ_FAIL,
            bot_constants.ERROR_MSG_ATTACHMENT_UNSUPPORTED,
            bot_constants.ERROR_MSG_GEMINI_API_ERROR,
            bot_constants.ERROR_MSG_GEMINI_INVALID_ARG,
            bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED,
            bot_constants.ERROR_MSG_GEMINI_BLOCKED_PROMPT,
            bot_constants.ERROR_MSG_GEMINI_BLOCKED_RESPONSE,
            bot_constants.ERROR_MSG_GEMINI_UNKNOWN,
            bot_constants.ERROR_MSG_INTERNAL,
            bot_constants.ERROR_MSG_PERMISSION_DENIED,
            bot_constants.ERROR_MSG_HISTORY_READ_FAIL,
            bot_constants.ERROR_MSG_LOWLOAD_UNAVAILABLE,
            bot_constants.ERROR_MSG_DEEP_CACHE_FAIL,
            bot_constants.ERROR_MSG_COMMAND_FORMAT,
            bot_constants.ERROR_MSG_POLL_INVALID,
            bot_constants.ERROR_MSG_TIMER_INVALID,
            bot_constants.ERROR_MSG_NO_CONTENT,
            bot_constants.ERROR_MSG_FILE_SIZE_LIMIT,
            bot_constants.ERROR_MSG_BUTTON_ERROR,
            bot_constants.ERROR_MSG_CHANNEL_ERROR,
            "Error", "Failed", "Block", "Limit", "Problem", "Cannot", "Invalid",
            "Rate", "Access", "Permission", "Unavailable", "Connection", "Unsupported"
        ]
        if text is None: return True
        return any(keyword in text for keyword in error_keywords)