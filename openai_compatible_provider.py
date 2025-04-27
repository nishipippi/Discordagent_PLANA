# openai_compatible_provider.py
# (OpenAI互換API用プロバイダーの実装 - Mistral含む)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionContentPartParam
from openai import APIStatusError, AuthenticationError, RateLimitError, APIConnectionError, InternalServerError
import asyncio
import base64
from typing import List, Dict, Any, Optional, Tuple, Union, Literal

from llm_provider import (
    LLMProvider, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_INVALID_ARGUMENT,
    ERROR_TYPE_BLOCKED_PROMPT, ERROR_TYPE_BLOCKED_RESPONSE,
    ERROR_TYPE_API_ERROR, ERROR_TYPE_UNKNOWN, ERROR_TYPE_INTERNAL,
    ERROR_TYPE_UNSUPPORTED_FEATURE
)
import bot_constants # エラーメッセージ定数を別ファイルからimport想定

# --- OpenAI互換 API エラーマッピング ---
def _map_openai_exception_to_error_type(e: Exception) -> Tuple[str, Optional[str]]:
    """OpenAI SDKの例外を共通エラータイプと詳細にマッピング"""
    if isinstance(e, RateLimitError):
        return ERROR_TYPE_RATE_LIMIT, str(e)
    elif isinstance(e, AuthenticationError):
        return ERROR_TYPE_API_ERROR, "Authentication failed (Invalid API Key)."
    elif isinstance(e, APIStatusError): # Other API errors with status codes
        status_code = e.status_code
        message = e.response.text # エラーレスポンスの本文を取得
        if not message: message = str(e)
        if status_code in [400, 422]: # Bad Request or Unprocessable Entity
             if "prompt is too long" in message.lower() or "context length" in message.lower():
                 return ERROR_TYPE_INVALID_ARGUMENT, "Input too large (text or image)."
             if "invalid base64" in message.lower():
                 return ERROR_TYPE_INVALID_ARGUMENT, "Invalid image data."
             return ERROR_TYPE_INVALID_ARGUMENT, message
        elif status_code == 403: # Forbidden
            if "blocked" in message.lower() or "filtered" in message.lower():
                 return ERROR_TYPE_BLOCKED_RESPONSE, message
            return ERROR_TYPE_API_ERROR, f"Access forbidden ({message})."
        elif status_code >= 500: # Server errors
            return ERROR_TYPE_API_ERROR, f"API server error ({status_code}): {message}"
        else: # Other 4xx errors
            return ERROR_TYPE_API_ERROR, f"API error ({status_code}): {message}"
    elif isinstance(e, APIConnectionError):
        return ERROR_TYPE_API_ERROR, f"Connection error: {e.message}"
    elif isinstance(e, InternalServerError):
         return ERROR_TYPE_API_ERROR, f"Internal server error from API: {e.message}"
    else:
        return ERROR_TYPE_UNKNOWN, str(e)

class OpenAICompatibleProvider(LLMProvider):
    SUPPORTED_IMAGE_DETAIL = "auto"
    VISION_MODEL_KEYWORDS = ["vision", "pixtral"]

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.primary_model_name: Optional[str] = None
        self.secondary_model_name: Optional[str] = None
        self.lowload_model_name: Optional[str] = None
        self.system_prompt: str = ""
        self.base_url: Optional[str] = None

    async def initialize(self, api_key: str, model_config: Dict[str, str], system_prompt: str, base_url: Optional[str] = None) -> bool:
        """OpenAICompatibleProviderの初期化"""
        if not base_url:
            print("Error: base_url is required for OpenAICompatibleProvider.")
            return False
        if not api_key:
             print("Error: api_key is missing for OpenAICompatibleProvider.")
             return False

        try:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self.system_prompt = system_prompt
            self.base_url = base_url

            self.primary_model_name = model_config.get('primary')
            self.secondary_model_name = model_config.get('secondary', self.primary_model_name)
            self.lowload_model_name = model_config.get('lowload')

            if not self.primary_model_name:
                print("Error: Primary model name is not configured for OpenAICompatibleProvider.")
                return False
            if not self.lowload_model_name:
                print("Warning: Lowload model name is not configured. Related features might be limited.")
            if not self.secondary_model_name:
                 self.secondary_model_name = self.primary_model_name

            return True
        except Exception as e:
            print(f"Exception within OpenAICompatibleProvider.initialize: {e}")
            self.client = None
            return False

    def _is_vision_model(self, model_name: Optional[str]) -> bool:
         """指定されたモデル名が画像対応モデルであるか簡易判定"""
         if not model_name: return False
         return any(keyword in model_name.lower() for keyword in self.VISION_MODEL_KEYWORDS)


    def _convert_history_to_openai_chat(
        self,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None,
        # target_model_is_vision: bool = False # この引数を削除
        target_model_name: Optional[str] = None # モデル名を引数として渡す
        ) -> List[ChatCompletionMessageParam]:
        """Gemini形式の履歴とパーツをOpenAI互換APIのChatMessageリストに変換"""
        messages: List[ChatCompletionMessageParam] = []

        # メソッド内でVision対応を判定
        target_model_is_vision = self._is_vision_model(target_model_name)

        if self.system_prompt:
             messages.append({"role": "system", "content": self.system_prompt})

        if deep_cache_summary:
            messages.append({"role": "user", "content": f"【長期記憶からの参考情報】\n{deep_cache_summary}"})

        if chat_history:
            for entry in chat_history:
                role = entry.get("role")
                parts = entry.get("parts", [])
                if not role or not parts: continue

                openai_role = "assistant" if role == "model" else "user"
                text_content = " ".join(p.get("text", "") for p in parts if "text" in p).strip()
                if text_content:
                     messages.append({"role": openai_role, "content": text_content})


        current_content: List[ChatCompletionContentPartParam] = []
        has_images_in_current_parts = False
        for part in content_parts:
             if "text" in part:
                 current_content.append({"type": "text", "text": part["text"]})
             elif "inline_data" in part:
                 has_images_in_current_parts = True
                 if target_model_is_vision:
                     try:
                         mime_type = part["inline_data"]["mime_type"]
                         data = part["inline_data"]["data"]
                         if isinstance(data, bytes):
                             b64_data = base64.b64encode(data).decode('utf-8')
                             image_url = f"data:{mime_type};base64,{b64_data}"
                             current_content.append({
                                 "type": "image_url",
                                 "image_url": {"url": image_url, "detail": self.SUPPORTED_IMAGE_DETAIL}
                             })
                         else: print("Warning: Invalid image data type in content_parts, skipping image processing.")
                     except Exception as e:
                         print(f"Warning: Failed to process image data for OpenAI Compatible API: {e}, skipping image.")
                 else:
                      # 画像非対応モデルの場合
                      print(f"Warning: Image detected but target model '{target_model_name}' ({target_model_is_vision}) might not support vision. Image will be ignored.")
                      # 画像パートを除去
                      # 注意: current_content を変更する場合、ループ内の後続の画像に影響する可能性があるため
                      # 新しいリストを作成するか、別の方法で処理する方が安全。
                      # ここでは簡易的に、画像パートを追加しないことで除去とする。
                      # current_content.append({"type": "text", "text": f"[Image ignored for {target_model_name}]"}) # 無視された旨を示すテキストを追加しても良い


        if current_content:
             messages.append({"role": "user", "content": current_content}) # type: ignore
        elif not messages:
             print("Warning: Message list is empty after processing all parts.")
             pass

        if messages and messages[-1]["role"] != "user":
             print(f"Warning: Last message role is not 'user': {messages[-1]['role']}. API might reject.")

        # 最後に、画像が content_parts に含まれていたが、ターゲットモデルが Vision 非対応で画像が除去された場合、
        # その旨を再度警告し、ログで確認できるようにする。
        # messages リストが生成された後で、画像パートが含まれているか最終確認する方が確実かもしれないが、
        # _convert_history_to_openai_chat の責任範囲を超えるため、ここでは画像の追加/除去の判定は上記のみとする。


        return messages


    async def _call_openai_api(
        self,
        model_name: str,
        messages: List[ChatCompletionMessageParam]
        ) -> Tuple[str, str]:
        """OpenAI互換 API (Mistral含む) を呼び出し、結果またはエラーメッセージを返す"""
        if not self.client: return model_name, self.format_error_message(ERROR_TYPE_INTERNAL, "API client not initialized.")
        if not model_name: return "No Model", self.format_error_message(ERROR_TYPE_INTERNAL, "Model name not specified.")
        if not messages: return model_name, self.format_error_message(ERROR_TYPE_INVALID_ARGUMENT, "No valid messages to send to API.")

        try:
            print(f"Calling OpenAI Compatible API ({model_name}, Base URL: {self.base_url})... Messages: {len(messages)}")
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
            )
            print(f"OpenAI Compatible API ({model_name}) response received.")

            if response.choices and response.choices[0].message:
                response_message = response.choices[0].message
                response_content = response_message.content
                finish_reason = response.choices[0].finish_reason

                if response_content is None: response_content = ""

                if finish_reason == "length":
                    print(f"Warning: Response truncated due to maximum length ({model_name}).")
                    error_type, error_detail = ERROR_TYPE_INVALID_ARGUMENT, "Output exceeded maximum token limit."
                    return model_name, response_content + f"\n\n...{self.format_error_message(error_type, error_detail)}"
                elif finish_reason == "stop": return model_name, response_content
                elif finish_reason == "tool_calls":
                     print(f"Warning: Tool call requested by model ({model_name}), but not handled.")
                     tool_calls = response_message.tool_calls
                     tool_info = ", ".join([f"{tc.function.name}({tc.function.arguments[:50]}...)" for tc in tool_calls]) if tool_calls else "N/A"
                     return model_name, response_content + f"\n\n(Tool call detected: {tool_info})"
                else:
                     print(f"Warning: Response finished with reason: {finish_reason} ({model_name}).")
                     if finish_reason == "content_filter":
                          return model_name, self.format_error_message(ERROR_TYPE_BLOCKED_RESPONSE, "Content filter triggered.")
                     return model_name, self.format_error_message(ERROR_TYPE_UNKNOWN, f"Finished with reason: {finish_reason}")
            else:
                print(f"Warning: No choices or message content in API response ({model_name}). Full Response: {response}")
                return model_name, self.format_error_message(ERROR_TYPE_UNKNOWN, "Empty response from API.")

        except Exception as e:
            print(f"Error during OpenAI Compatible API call ({model_name}): {e}")
            error_type, error_detail = _map_openai_exception_to_error_type(e)
            raise e


    async def generate_response(
        self,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None,
    ) -> Tuple[str, str]:

        primary_model_name = self.get_model_name('primary')
        secondary_model_name = self.get_model_name('secondary')

        # 1. プライマリモデルで試行
        if primary_model_name:
            current_model_to_try = primary_model_name
            try:
                # _convert_history_to_openai_chat にモデル名を渡す
                api_messages = self._convert_history_to_openai_chat(
                    content_parts, chat_history, deep_cache_summary, target_model_name=current_model_to_try
                )
                if not api_messages:
                     print("Warning: No valid messages generated for API call (primary).")
                     return current_model_to_try, self.format_error_message(ERROR_TYPE_INVALID_ARGUMENT, "No content to send to API (primary).")

                return await self._call_openai_api(current_model_to_try, api_messages)

            except Exception as e:
                error_type, error_detail = _map_openai_exception_to_error_type(e)
                if self.is_rate_limit_error(e) and secondary_model_name and secondary_model_name != primary_model_name:
                    print(f"Primary model ({primary_model_name}) rate limited. Falling back to secondary ({secondary_model_name}).")
                    pass # フォールバック処理へ
                else:
                    print(f"Error with Primary model ({primary_model_name}): {e}. No fallback possible or error is not rate limit.")
                    return primary_model_name, self.format_error_message(error_type, error_detail)
        else:
             print("Error: Primary model name not set.")
             pass # フォールバックを試みる


        # 2. セカンダリモデルで試行 (プライマリ失敗時)
        if secondary_model_name:
             current_model_to_try = secondary_model_name
             print(f"Attempting fallback with Secondary model ({current_model_to_try})...")

             try:
                 # _convert_history_to_openai_chat にセカンダリモデル名を渡す
                 # これにより、セカンダリがVision非対応なら画像が除去される
                 secondary_api_messages = self._convert_history_to_openai_chat(
                      content_parts, chat_history, deep_cache_summary, target_model_name=current_model_to_try
                  )
                 if not secondary_api_messages:
                      print("Warning: No valid messages generated for API call (secondary).")
                      return current_model_to_try, self.format_error_message(ERROR_TYPE_INVALID_ARGUMENT, "No content to send to API (secondary).")

                 return await self._call_openai_api(current_model_to_try, secondary_api_messages)

             except Exception as e:
                 print(f"Error with Secondary model ({current_model_to_try}): {e}")
                 error_type, error_detail = _map_openai_exception_to_error_type(e)
                 return current_model_to_try, self.format_error_message(error_type, error_detail)
        else:
            print("Error: No secondary model available for fallback.")
            # プライマリのエラーメッセージを返す (上で既に返されているはずだが念のため)
            primary_error_msg = self.format_error_message(ERROR_TYPE_INTERNAL, "Primary model failed and no secondary model available.")
            return primary_model_name or "No Model", primary_error_msg


    async def generate_lowload_response(self, prompt: str) -> Optional[str]:
        lowload_model_name = self.get_model_name('lowload')
        if not self.client or not lowload_model_name:
            print("Warning: Lowload model/client is not available.")
            return None

        # Lowload モデル用のメッセージリスト作成
        # _convert_history_to_openai_chat を Lowload モデル名付きで呼び出す
        try:
             messages = self._convert_history_to_openai_chat(
                  content_parts=[{'text': prompt}], # 単純なテキストプロンプトとして渡す
                  chat_history=None, # Lowloadには履歴やDeepCacheは渡さない想定
                  deep_cache_summary=None,
                  target_model_name=lowload_model_name # Lowloadモデル名を渡す
              )
             if not messages:
                  print("Warning: No valid messages generated for Lowload API call.")
                  return None

        except Exception as e:
             print(f"Error converting prompt to Lowload API format: {e}")
             return None


        try:
            model_name, response_text = await self._call_openai_api(lowload_model_name, messages)
            if self._is_error_message(response_text):
                 print(f"Lowload response generated an error message: {response_text}")
                 return None
            return response_text
        except Exception as e:
             print(f"Lowload API call failed for model {lowload_model_name}: {e}")
             return None


    def format_error_message(self, error_type: str, detail: Optional[str] = None) -> str:
        if error_type == ERROR_TYPE_RATE_LIMIT: return bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED
        elif error_type == ERROR_TYPE_INVALID_ARGUMENT:
            if detail and "Input too large" in detail: return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG + " (入力過大)"
            elif detail and "token limit" in detail: return bot_constants.ERROR_MSG_MAX_TEXT_SIZE
            elif detail and "Invalid image data" in detail: return bot_constants.ERROR_MSG_IMAGE_READ_FAIL + " (データ破損)"
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
        elif error_type == ERROR_TYPE_UNSUPPORTED_FEATURE: return bot_constants.ERROR_MSG_INTERNAL + " (未対応機能)"
        elif error_type == ERROR_TYPE_UNKNOWN:
            if detail: return bot_constants.ERROR_MSG_GEMINI_UNKNOWN + f" ({detail[:50]}...)"
            return bot_constants.ERROR_MSG_GEMINI_UNKNOWN
        else: return bot_constants.ERROR_MSG_GEMINI_UNKNOWN

    def is_rate_limit_error(self, exception: Exception) -> bool:
        return isinstance(exception, RateLimitError) or (isinstance(exception, APIStatusError) and exception.status_code == 429)

    def is_invalid_argument_error(self, exception: Exception) -> bool:
        return isinstance(exception, APIStatusError) and exception.status_code in [400, 422]

    def get_model_name(self, model_type: Literal["primary", "secondary", "lowload"]) -> Optional[str]:
        """OpenAI互換モデル名を取得"""
        if model_type == "primary": return self.primary_model_name
        elif model_type == "secondary": return self.secondary_model_name
        elif model_type == "lowload": return self.lowload_model_name
        return None