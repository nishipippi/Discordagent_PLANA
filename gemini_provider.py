# gemini_provider.py
# (Gemini API用プロバイダーの実装)

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, GoogleAPIError
import asyncio
import base64
import re
from typing import List, Dict, Any, Optional, Tuple, Literal # Literal をインポートに追加

from llm_provider import (
    LLMProvider, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_INVALID_ARGUMENT,
    ERROR_TYPE_BLOCKED_PROMPT, ERROR_TYPE_BLOCKED_RESPONSE,
    ERROR_TYPE_API_ERROR, ERROR_TYPE_UNKNOWN, ERROR_TYPE_INTERNAL,
    ERROR_TYPE_UNSUPPORTED_FEATURE
)
import bot_constants # エラーメッセージ定数を別ファイルからimport想定

# --- Gemini エラーマッピング ---
def _map_gemini_exception_to_error_type(e: Exception) -> Tuple[str, Optional[str]]:
    """Gemini SDKの例外を共通エラータイプと詳細にマッピング"""
    if isinstance(e, ResourceExhausted):
        return ERROR_TYPE_RATE_LIMIT, str(e)
    elif isinstance(e, InvalidArgument):
        detail = str(e)
        if "Unsupported MIME type" in detail:
            match = re.search(r"Unsupported MIME type: (.*?)\.", detail)
            mime_type_error = match.group(1) if match else "不明"
            return ERROR_TYPE_INVALID_ARGUMENT, f"Unsupported MIME type found in request ({mime_type_error})."
        elif "prompt is too long" in detail.lower() or "413" in detail:
             return ERROR_TYPE_INVALID_ARGUMENT, "Input too large (text or image)."
        return ERROR_TYPE_INVALID_ARGUMENT, detail
    elif isinstance(e, GoogleAPIError): # 他のGoogle APIエラー
        return ERROR_TYPE_API_ERROR, str(e)
    else:
        return ERROR_TYPE_UNKNOWN, str(e)

def _map_gemini_finish_reason_to_error(finish_reason: str, response: Any) -> Optional[Tuple[str, Optional[str]]]:
    """Geminiの応答のfinish_reasonを共通エラータイプと詳細にマッピング"""
    if finish_reason == "SAFETY":
        block_detail = "Safety settings triggered."
        if hasattr(response, 'candidates') and response.candidates:
             candidate = response.candidates[0]
             if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                 blocked_categories = [r.category.name for r in candidate.safety_ratings if r.probability.name not in ["NEGLIGIBLE", "LOW"]]
                 if blocked_categories:
                     block_detail += f" Categories: {', '.join(blocked_categories)}"
        return ERROR_TYPE_BLOCKED_RESPONSE, block_detail
    elif finish_reason == "MAX_TOKENS":
        return ERROR_TYPE_INVALID_ARGUMENT, "Output exceeded maximum token limit."
    elif finish_reason in ["RECITATION", "OTHER"]:
        return ERROR_TYPE_UNKNOWN, f"Stopped due to reason: {finish_reason}"
    return None

def _map_gemini_prompt_feedback_to_error(response: Any) -> Optional[Tuple[str, Optional[str]]]:
    """Geminiのprompt_feedbackを共通エラータイプと詳細にマッピング"""
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
        reason = response.prompt_feedback.block_reason.name
        return ERROR_TYPE_BLOCKED_PROMPT, f"Prompt blocked due to reason: {reason}"
    return None


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.client = None
        self.primary_model: Optional[genai.GenerativeModel] = None
        self.secondary_model: Optional[genai.GenerativeModel] = None
        self.lowload_model: Optional[genai.GenerativeModel] = None
        self.system_prompt = ""

    async def initialize(self, api_key: str, model_config: Dict[str, str], system_prompt: str, base_url: Optional[str] = None) -> bool:
        """GeminiProviderの初期化 (base_urlは無視)"""
        try:
            genai.configure(api_key=api_key)
            self.client = genai
            self.system_prompt = system_prompt
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            primary_model_name = model_config.get('primary')
            secondary_model_name = model_config.get('secondary')
            lowload_model_name = model_config.get('lowload')

            if primary_model_name:
                try: self.primary_model = self.client.GenerativeModel(model_name=primary_model_name, system_instruction=self.system_prompt, safety_settings=safety_settings)
                except Exception as e: print(f"Warning: Failed to initialize Gemini Primary Model ({primary_model_name}): {e}"); self.primary_model = None
            if secondary_model_name:
                try: self.secondary_model = self.client.GenerativeModel(model_name=secondary_model_name, system_instruction=self.system_prompt, safety_settings=safety_settings)
                except Exception as e: print(f"Warning: Failed to initialize Gemini Secondary Model ({secondary_model_name}): {e}"); self.secondary_model = None
            if lowload_model_name:
                try: self.lowload_model = self.client.GenerativeModel(model_name=lowload_model_name, safety_settings=safety_settings)
                except Exception as e: print(f"Warning: Failed to initialize Gemini Lowload Model ({lowload_model_name}): {e}"); self.lowload_model = None

            if not self.primary_model and not self.secondary_model: print("Error: No primary or secondary Gemini model could be initialized."); return False
            if not self.lowload_model: print("Warning: Gemini Lowload model unavailable. Related features might be limited."); # 初期化は成功とみなす

            return True
        except Exception as e:
            print(f"Exception within GeminiProvider.initialize: {e}")
            return False

    async def _generate_content_internal(
        self,
        model: genai.GenerativeModel,
        model_name_for_log: str,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None # chat_history は llm_provider._format_history_for_prompt で整形済みテキストを想定
    ) -> Tuple[str, str]:
        """Gemini API呼び出しのコアロジック"""
        if not model: return model_name_for_log, self.format_error_message(ERROR_TYPE_INTERNAL, "Model not initialized.")

        # chat_history は _format_history_for_prompt で整形されたテキスト文字列
        # Gemini の generate_content は履歴を ChatMessage のリスト形式で受け取るため、再構築が必要
        # または、システム指示として渡す、最初のUserメッセージに含めるなど工夫が必要。
        # ここでは、簡易的に chat_history テキストを最初のユーザーメッセージに含める。
        # より正確な履歴対話には、chat_history が [{'role': 'user'/'model', 'parts': [...]}] のリスト形式で渡される必要がある。
        # LLMProvider の generate_response の chat_history の型ヒント定義と実際の渡し方が一致していない可能性がある。
        # LLMProvider の generate_response の chat_history は [{'role': 'user'/'model', 'parts': [...]}] リスト形式のはずなので、その前提で修正。

        gemini_history_messages: List[Dict[str, Any]] = []
        if chat_history:
             # 渡される chat_history は [{'role': 'user'/'model', 'parts': [...]}] のリスト形式であることを前提とする
             gemini_history_messages = chat_history

        gemini_parts = content_parts # 現在のユーザー入力パーツ

        # Deep Cache は generate_response で既に最初のユーザーメッセージに付加されているか、
        # システム命令に含まれていることを想定。ここではそのままAPIに渡す。

        try:
            print(f"Calling Gemini API ({model_name_for_log})... History Messages: {len(gemini_history_messages) if gemini_history_messages else 0}, Current Parts: {len(gemini_parts)}")
            # 履歴がある場合は start_chat + send_message
            if gemini_history_messages:
                # 履歴の最後のロールを確認し、現在の入力 (User) と交互になっているか確認
                last_role = gemini_history_messages[-1]['role'] if gemini_history_messages else None
                if last_role == 'user':
                     # 履歴の最後がユーザーなので、本来はモデルの応答が期待されるが、
                     # APIによってはユーザー -> ユーザーのような連続も許容する場合がある
                     # ここではそのまま履歴として渡す。必要であればエラー処理や調整を追加。
                     print("Warning: Chat history ends with 'user' role. Appending new user message.")
                elif last_role == 'model' or last_role is None:
                     # 履歴の最後がモデルまたは履歴が空なので、ユーザー入力で開始/継続はOK
                     pass
                # ChatSessionを開始し、履歴と現在のユーザー入力パーツを渡す
                chat = model.start_chat(history=gemini_history_messages)
                response = await asyncio.to_thread(chat.send_message, gemini_parts, stream=False)
            else:
                # 履歴がない場合は generate_content (最初のユーザー入力)
                response = await asyncio.to_thread(model.generate_content, gemini_parts, stream=False)
            print(f"Gemini API ({model_name_for_log}) response received.")

            response_text = None; finish_reason = "UNKNOWN"; error_type = None; error_detail = None
            prompt_error = _map_gemini_prompt_feedback_to_error(response)
            if prompt_error: error_type, error_detail = prompt_error; print(f"Warning: Prompt blocked by Gemini ({model_name_for_log}). Reason: {error_detail}"); return model_name_for_log, self.format_error_message(error_type, error_detail)

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0];
                if candidate.finish_reason: finish_reason = candidate.finish_reason.name
                if candidate.content and candidate.content.parts: response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                finish_error = _map_gemini_finish_reason_to_error(finish_reason, response)
                if finish_error:
                    error_type, error_detail = finish_error; print(f"Warning: Response generation stopped by Gemini ({model_name_for_log}). Reason: {finish_reason}, Detail: {error_detail}")
                    if error_type == ERROR_TYPE_INVALID_ARGUMENT and "token limit" in (error_detail or "") and response_text: return model_name_for_log, response_text + f"\n\n...{self.format_error_message(error_type, error_detail)}"
                    return model_name_for_log, self.format_error_message(error_type, error_detail)
            elif hasattr(response, 'text'): response_text = response.text; finish_reason = "COMPLETED"

            if response_text is not None: return model_name_for_log, response_text
            else: print(f"Warning: No text content in Gemini response ({model_name_for_log}). Finish Reason: {finish_reason}"); return model_name_for_log, self.format_error_message(ERROR_TYPE_UNKNOWN, f"No text content received. Finish Reason: {finish_reason}")

        except Exception as e:
            print(f"Error during Gemini API call ({model_name_for_log}): {e}")
            error_type, error_detail = _map_gemini_exception_to_error_type(e)
            raise e

    async def generate_response(
        self,
        content_parts: List[Dict[str, Any]], # 現在のユーザー入力パーツ
        chat_history: Optional[List[Dict[str, Any]]] = None, # 過去の会話履歴リスト [{'role': ..., 'parts': [...]}]
        deep_cache_summary: Optional[str] = None, # Deep Cacheサマリーテキスト
    ) -> Tuple[str, str]:
        # Deep Cache を最初のユーザーメッセージに含める処理は bot.py 側で行う
        # ここでは、渡された chat_history (LLM形式リスト) と content_parts をそのままAPIに渡す

        primary_model_name = self.get_model_name('primary') or "N/A"
        secondary_model_name = self.get_model_name('secondary') or "N/A"

        if self.primary_model:
            try:
                return await self._generate_content_internal(self.primary_model, primary_model_name, content_parts, chat_history)
            except Exception as e:
                error_type, error_detail = _map_gemini_exception_to_error_type(e)
                if self.is_rate_limit_error(e) and self.secondary_model:
                    print(f"Gemini Primary model ({primary_model_name}) rate limited. Falling back to secondary ({secondary_model_name}).")
                    pass
                else:
                    print(f"Error with Gemini Primary model ({primary_model_name}): {e}. No fallback possible or error is not rate limit.")
                    return primary_model_name, self.format_error_message(error_type, error_detail)
        else:
             print("Gemini Primary model not available. Attempting Secondary model.")

        if self.secondary_model:
            try:
                return await self._generate_content_internal(self.secondary_model, secondary_model_name, content_parts, chat_history)
            except Exception as e:
                print(f"Error with Gemini Secondary model ({secondary_model_name}): {e}")
                error_type, error_detail = _map_gemini_exception_to_error_type(e)
                return secondary_model_name, self.format_error_message(error_type, error_detail)
        else:
            print("Error: Both primary and secondary Gemini models are unavailable or failed.")
            return "No Model", self.format_error_message(ERROR_TYPE_INTERNAL, "No available Gemini models.")


    async def generate_lowload_response(self, prompt: str) -> Optional[str]:
        # (変更なし - promptテキストのみを generate_content に渡す)
        if not self.lowload_model:
            print("Warning: Gemini Lowload model is not available.")
            return None
        model_name = self.get_model_name('lowload') or "N/A Lowload"
        try:
            print(f"Calling Gemini Lowload API ({model_name})...")
            response = await asyncio.to_thread(self.lowload_model.generate_content, [{'text': prompt}], stream=False)
            print(f"Gemini Lowload API ({model_name}) response received.")
            response_text = None; finish_reason = "UNKNOWN"; prompt_error = _map_gemini_prompt_feedback_to_error(response)
            if prompt_error: error_type, error_detail = prompt_error; print(f"Warning: Lowload prompt blocked ({model_name}). Reason: {error_detail}"); return None
            if hasattr(response, 'candidates') and response.candidates:
                 candidate = response.candidates[0]; 
                 if candidate.finish_reason: finish_reason = candidate.finish_reason.name
                 if candidate.content and candidate.content.parts: response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                 finish_error = _map_gemini_finish_reason_to_error(finish_reason, response)
                 if finish_error:
                      error_type, error_detail = finish_error; print(f"Warning: Lowload response generation stopped ({model_name}). Reason: {finish_reason}, Detail: {error_detail}")
                      if error_type == ERROR_TYPE_INVALID_ARGUMENT and "token limit" in (error_detail or "") and response_text: return response_text.strip()
                      return None
            elif hasattr(response, 'text'): response_text = response.text; finish_reason = "COMPLETED"
            if response_text is not None: return response_text.strip()
            else: print(f"Warning: No text content in Gemini Lowload response ({model_name}). Finish Reason: {finish_reason}"); return None
        except Exception as e:
            print(f"Error during Gemini Lowload API call ({model_name}): {e}"); return None

    def format_error_message(self, error_type: str, detail: Optional[str] = None) -> str:
        if error_type == ERROR_TYPE_RATE_LIMIT: return bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED
        elif error_type == ERROR_TYPE_INVALID_ARGUMENT:
            if detail and "Unsupported MIME type" in detail: match = re.search(r"Unsupported MIME type.*?\((.*?)\)", detail); mime_type = match.group(1) if match else "不明"; return bot_constants.ERROR_MSG_ATTACHMENT_UNSUPPORTED + f" ({mime_type})"
            elif detail and "Input too large" in detail: return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG + " (入力過大)"
            elif detail and "token limit" in detail: return bot_constants.ERROR_MSG_MAX_TEXT_SIZE
            return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG
        elif error_type == ERROR_TYPE_BLOCKED_PROMPT: return bot_constants.ERROR_MSG_GEMINI_BLOCKED_PROMPT
        elif error_type == ERROR_TYPE_BLOCKED_RESPONSE: return bot_constants.ERROR_MSG_GEMINI_BLOCKED_RESPONSE
        elif error_type == ERROR_TYPE_API_ERROR:
             if detail and "Authentication failed" in detail: return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (認証失敗)"
             if detail and "Connection error" in detail: return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (接続失敗)"
             if detail and "server error" in detail: return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (サーバーエラー)"
             if detail: return bot_constants.ERROR_MSG_GEMINI_API_ERROR + f" ({detail[:50]}...)"
             return bot_constants.ERROR_MSG_GEMINI_API_ERROR
        elif error_type == ERROR_TYPE_INTERNAL: return bot_constants.ERROR_MSG_INTERNAL
        elif error_type == ERROR_TYPE_UNKNOWN:
            if detail: return bot_constants.ERROR_MSG_GEMINI_UNKNOWN + f" ({detail[:50]}...)"
            return bot_constants.ERROR_MSG_GEMINI_UNKNOWN
        elif error_type == ERROR_TYPE_UNSUPPORTED_FEATURE: return bot_constants.ERROR_MSG_INTERNAL + " (未対応機能)"
        else: return bot_constants.ERROR_MSG_GEMINI_UNKNOWN

    def is_rate_limit_error(self, exception: Exception) -> bool: return isinstance(exception, ResourceExhausted)
    def is_invalid_argument_error(self, exception: Exception) -> bool: return isinstance(exception, InvalidArgument)
    def get_model_name(self, model_type: Literal["primary", "secondary", "lowload"]) -> Optional[str]:
        if model_type == "primary" and self.primary_model: return self.primary_model.model_name
        elif model_type == "secondary" and self.secondary_model: return self.secondary_model.model_name
        elif model_type == "lowload" and self.lowload_model: return self.lowload_model.model_name
        return None