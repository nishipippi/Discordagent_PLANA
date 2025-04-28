# llm_manager.py
# (LLMプロバイダーの初期化、管理、切り替え)

import config
import bot_constants
from llm_provider import (
    LLMProvider, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_INVALID_ARGUMENT,
    ERROR_TYPE_BLOCKED_PROMPT, ERROR_TYPE_BLOCKED_RESPONSE,
    ERROR_TYPE_API_ERROR, ERROR_TYPE_INTERNAL, ERROR_TYPE_UNKNOWN,
    ERROR_TYPE_UNSUPPORTED_FEATURE
)
from gemini_provider import GeminiProvider
from openai_compatible_provider import OpenAICompatibleProvider
from typing import Optional, Dict, Tuple, List, Any, Literal # Literalを追加

# --- グローバル変数 (モジュールレベルで管理) ---
_gemini_handler: Optional[GeminiProvider] = None
_mistral_handler: Optional[OpenAICompatibleProvider] = None
_llm_handler: Optional[LLMProvider] = None
_current_provider_name: str = config.INITIAL_LLM_PROVIDER_NAME
_persona_instruction: str = "" # on_readyで設定

def set_persona_instruction(instruction: str):
    """ペルソナ指示を設定"""
    global _persona_instruction
    _persona_instruction = instruction

def get_persona_instruction() -> str:
    """ペルソナ指示を取得"""
    return _persona_instruction

def get_current_provider() -> Optional[LLMProvider]:
    """現在アクティブなLLMプロバイダーインスタンスを取得"""
    return _llm_handler

def get_current_provider_name() -> str:
    """現在アクティブなLLMプロバイダー名を取得"""
    return _current_provider_name

def get_active_model_name(model_type: Literal["primary", "secondary", "lowload"]) -> Optional[str]:
    """現在アクティブなプロバイダーの指定されたモデル名を取得"""
    if _llm_handler:
        return _llm_handler.get_model_name(model_type)
    return None

async def initialize_provider(provider_name: str = config.INITIAL_LLM_PROVIDER_NAME, force_reinitialize: bool = False) -> Optional[LLMProvider]:
    """指定された名前のLLMプロバイダーを初期化または取得する"""
    global _gemini_handler, _mistral_handler, _llm_handler, _current_provider_name, _persona_instruction

    provider_name = provider_name.upper()
    provider_display_name = provider_name.capitalize()

    # ペルソナが未設定ならデフォルトを設定 (client_idは後で設定される想定)
    if not _persona_instruction:
        _persona_instruction = bot_constants.PERSONA_TEMPLATE.format(client_id="ボット")

    # 既存ハンドラの確認と再利用
    existing_handler: Optional[LLMProvider] = None
    if provider_name == 'GEMINI':
        existing_handler = _gemini_handler
    elif provider_name == 'MISTRAL':
        existing_handler = _mistral_handler

    if existing_handler and not force_reinitialize:
        print(f"{provider_display_name} provider already initialized.")
        # 現在のハンドラーを更新
        _llm_handler = existing_handler
        _current_provider_name = provider_name
        return existing_handler

    print(f"Attempting to initialize {provider_display_name} provider...")

    provider: Optional[LLMProvider] = None
    api_key: Optional[str] = None
    model_config: Dict[str, Optional[str]] = {} # Optional[str] を許容するように変更
    base_url: Optional[str] = None

    if provider_name == 'GEMINI':
        api_key = config.GEMINI_API_KEY
        model_config = config.GEMINI_MODEL_CONFIG
        if not api_key:
            print(f"Warning: {provider_display_name} API Key (GEMINI_API_KEY) not found.")
            return None
        provider = GeminiProvider()
        # 古いハンドラーをクリア (再初期化の場合)
        if force_reinitialize: _gemini_handler = None

    elif provider_name == 'MISTRAL':
        api_key = config.MISTRAL_API_KEY
        model_config = config.MISTRAL_MODEL_CONFIG
        base_url = config.MISTRAL_API_BASE_URL
        if not api_key:
            print(f"Warning: {provider_display_name} API Key (MISTRAL_API_KEY) not found.")
            return None
        if not base_url:
             print(f"Warning: {provider_display_name} API Base URL (MISTRAL_API_BASE_URL) not found.")
             return None
        provider = OpenAICompatibleProvider()
        # 古いハンドラーをクリア (再初期化の場合)
        if force_reinitialize: _mistral_handler = None

    else:
        print(f"Error: Unknown provider name '{provider_name}' requested.")
        return None

    try:
        # model_config の値が None でないことを確認して渡す (initialize メソッドの型ヒントに合わせる)
        valid_model_config: Dict[str, str] = {k: v for k, v in model_config.items() if v is not None}

        initialized = await provider.initialize(
            api_key=api_key,
            model_config=valid_model_config, # None を含まない辞書を渡す
            system_prompt=_persona_instruction,
            base_url=base_url
        )
        if initialized:
            print(f"{provider_display_name} provider initialized successfully.")
            # グローバル変数に格納
            if provider_name == 'GEMINI':
                _gemini_handler = provider
            elif provider_name == 'MISTRAL':
                _mistral_handler = provider
            # 現在のハンドラーを更新
            _llm_handler = provider
            _current_provider_name = provider_name
            return provider
        else:
            print(f"Error: {provider_display_name} provider initialization failed.")
            # 初期化失敗時に対応するハンドラー変数をNoneにする
            if provider_name == 'GEMINI': _gemini_handler = None
            elif provider_name == 'MISTRAL': _mistral_handler = None
            # もし現在アクティブなハンドラーが失敗したものなら、それもNoneにする
            if _llm_handler == provider: _llm_handler = None
            return None
    except Exception as e:
        print(f"CRITICAL: Exception during {provider_display_name} provider initialization: {e}")
        import traceback
        traceback.print_exc()
        if provider_name == 'GEMINI': _gemini_handler = None
        elif provider_name == 'MISTRAL': _mistral_handler = None
        if _llm_handler == provider: _llm_handler = None
        return None

async def switch_provider(target_provider_name: str) -> Tuple[bool, str]:
    """LLMプロバイダーを切り替える"""
    global _llm_handler, _current_provider_name

    target_provider_name = target_provider_name.upper()
    if target_provider_name == _current_provider_name:
        return False, f"既に {target_provider_name.capitalize()} プロバイダーを使用中です。"

    print(f"Attempting to switch to {target_provider_name} provider.")
    new_handler = await initialize_provider(target_provider_name) # 初期化または既存ハンドラ取得

    if new_handler:
        _llm_handler = new_handler
        _current_provider_name = target_provider_name
        print(f"Successfully switched provider to {_current_provider_name}.")
        return True, f"プロバイダーを {_current_provider_name.capitalize()} に切り替えました。"
    else:
        # 切り替え失敗時は元のプロバイダーを維持しようと試みる（もしあれば）
        print(f"Failed to initialize or switch to {target_provider_name}. Keeping current provider: {_current_provider_name}")
        # 元のプロバイダーを再設定しようとする (もしNoneになっていたら初期化試行)
        if _current_provider_name:
             current_handler_still_valid = await initialize_provider(_current_provider_name)
             if not current_handler_still_valid:
                  print(f"Warning: Failed to re-validate the current provider '{_current_provider_name}' after switch failure.")
                  _llm_handler = None # 現在のハンドラも無効と判断
        else:
             _llm_handler = None # 元のプロバイダ名すらない場合

        return False, f"{target_provider_name.capitalize()} プロバイダーの初期化/利用に失敗しました。APIキーや設定を確認してください。"

async def generate_response(
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None
    ) -> Tuple[str, str]:
    """現在アクティブなプロバイダーで応答を生成する"""
    if not _llm_handler:
        return "No Provider", bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not initialized)"
    try:
        return await _llm_handler.generate_response(content_parts, chat_history, deep_cache_summary)
    except Exception as e:
        print(f"Error during generate_response with {_current_provider_name}: {e}")
        # _llm_handler.format_error_message が利用可能か確認
        if hasattr(_llm_handler, 'format_error_message'):
            # エラーの種類を特定しようと試みる (簡易的)
            error_type = ERROR_TYPE_UNKNOWN
            if _llm_handler.is_rate_limit_error(e): error_type = ERROR_TYPE_RATE_LIMIT
            elif _llm_handler.is_invalid_argument_error(e): error_type = ERROR_TYPE_INVALID_ARGUMENT
            # 他の具体的なエラータイプへのマッピングは、ここでは難しい場合がある
            return _current_provider_name, _llm_handler.format_error_message(error_type, str(e))
        else:
            # format_error_message がない場合 (予期せぬ状況)
            return _current_provider_name, bot_constants.ERROR_MSG_INTERNAL + f" (Error formatting message: {e})"


async def generate_lowload_response(prompt: str) -> Optional[str]:
    """現在アクティブなプロバイダーの低負荷モデルで応答を生成する"""
    if not _llm_handler:
        print("Warning: LLM Provider not initialized for lowload response.")
        return None
    try:
        return await _llm_handler.generate_lowload_response(prompt)
    except Exception as e:
        print(f"Error during generate_lowload_response with {_current_provider_name}: {e}")
        return None # Lowloadではエラー詳細は返さない

def is_error_message(text: Optional[str]) -> bool:
    """応答テキストがエラーメッセージか判定 (共通ヘルパー)"""
    if text is None: return True
    if not _llm_handler: return False # ハンドラがない場合は判定不能だが、Falseを返す
    # プロバイダー固有の判定メソッドがあればそれを使う (推奨)
    if hasattr(_llm_handler, '_is_error_message'):
         return _llm_handler._is_error_message(text)
    # なければ共通のキーワード判定 (llm_provider.py に定義済み)
    # llm_provider が import されていないため bot_constants を直接参照するか、判定ロジックをここに持つ
    # ここでは llm_provider._is_error_message が期待されるが、Circular Import に注意
    # シンプルに bot_constants のエラーメッセージリストで判定
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
        # より汎用的なキーワードは bot_constants 側に移動させるか、ここで定義
        "Error", "Failed", "Block", "Limit", "Problem", "Cannot", "Invalid",
        "Rate", "Access", "Permission", "Unavailable", "Connection", "Unsupported"
    ]
    # 大文字小文字を区別しない判定
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in error_keywords)