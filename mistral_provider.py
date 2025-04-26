# mistral_provider.py
# (Mistral API用プロバイダーの実装 - Pixtral対応)

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, ToolCall
from mistralai.exceptions import MistralAPIException, MistralConnectionException, MistralException
import asyncio
import base64
from typing import List, Dict, Any, Optional, Tuple

from llm_provider import (
    LLMProvider, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_INVALID_ARGUMENT,
    ERROR_TYPE_BLOCKED_PROMPT, ERROR_TYPE_BLOCKED_RESPONSE, # Mistralはこれらを明示的に返さない可能性
    ERROR_TYPE_API_ERROR, ERROR_TYPE_UNKNOWN, ERROR_TYPE_INTERNAL,
    ERROR_TYPE_UNSUPPORTED_FEATURE
)
import bot_constants # エラーメッセージ定数を別ファイルからimport想定

# --- Mistral エラーマッピング ---
def _map_mistral_exception_to_error_type(e: Exception) -> Tuple[str, Optional[str]]:
    """Mistral SDKの例外を共通エラータイプと詳細にマッピング"""
    if isinstance(e, MistralAPIException):
        status_code = e.status_code
        message = e.message
        if status_code == 429:
            return ERROR_TYPE_RATE_LIMIT, message
        elif status_code in [400, 422]: # Bad Request or Unprocessable Entity
             # 詳細メッセージから判断
             if "prompt is too long" in message.lower() or "maximum context length" in message.lower():
                 return ERROR_TYPE_INVALID_ARGUMENT, "Input too large (text or image)."
             if "invalid base64" in message.lower():
                 return ERROR_TYPE_INVALID_ARGUMENT, "Invalid image data."
             # 他の引数エラーの可能性
             return ERROR_TYPE_INVALID_ARGUMENT, message
        elif status_code == 401: # Unauthorized
            return ERROR_TYPE_API_ERROR, "Authentication failed (Invalid API Key)."
        elif status_code == 403: # Forbidden (権限、安全フィルターなど?)
            # Mistralが安全フィルターでブロックした場合、403を返すことがある？要確認
            # 現状では汎用APIエラーとして扱う
            return ERROR_TYPE_API_ERROR, f"Access forbidden or potential content block ({message})."
        elif status_code >= 500: # Server errors
            return ERROR_TYPE_API_ERROR, f"Mistral server error ({status_code}): {message}"
        else: # その他の4xxエラー
            return ERROR_TYPE_API_ERROR, f"Mistral API error ({status_code}): {message}"
    elif isinstance(e, MistralConnectionException):
        return ERROR_TYPE_API_ERROR, f"Connection error: {e}"
    elif isinstance(e, MistralException): # SDKの他の基底エラー
        return ERROR_TYPE_UNKNOWN, str(e)
    else:
        return ERROR_TYPE_UNKNOWN, str(e)

class MistralProvider(LLMProvider):
    SUPPORTED_LOWLOAD_ROLES = {"user", "assistant"} # 低負荷モデルはSystem Prompt非対応の場合がある

    def __init__(self):
        self.client: Optional[MistralClient] = None
        self.primary_model_name: Optional[str] = None
        self.secondary_model_name: Optional[str] = None # Mistralは通常フォールバック不要だが一応保持
        self.lowload_model_name: Optional[str] = None
        self.system_prompt: str = ""

    async def initialize(self, api_key: str, model_config: Dict[str, str], system_prompt: str) -> bool:
        try:
            self.client = MistralClient(api_key=api_key)
            self.system_prompt = system_prompt
            self.primary_model_name = model_config.get('primary')
            self.secondary_model_name = model_config.get('secondary', self.primary_model_name) # デフォルトはプライマリと同じ
            self.lowload_model_name = model_config.get('lowload')

            # モデル名の存在チェック
            if not self.primary_model_name:
                print("Error: Mistral Primary model name is not configured.")
                return False
            if not self.lowload_model_name:
                print("Warning: Mistral Lowload model name is not configured. Related features might be limited.")
            if not self.secondary_model_name:
                 self.secondary_model_name = self.primary_model_name # フォールバック先をプライマリに設定

            # TODO: 必要ならここで `client.models.list()` を呼んでモデル存在確認？ (APIコール増える)
            print(f"MistralProvider initialized. Primary: {self.primary_model_name}, Secondary: {self.secondary_model_name}, Lowload: {self.lowload_model_name}")
            return True
        except Exception as e:
            print(f"Error initializing Mistral client: {e}")
            return False

    def _convert_history_to_mistral_chat(
        self,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None,
        target_model: Optional[str] = None # モデルによってSystem Promptの扱いを変える場合
        ) -> List[ChatMessage]:
        """Gemini形式の履歴とパーツをMistralのChatMessageリストに変換"""
        messages: List[ChatMessage] = []

        # 1. System Prompt (サポートされている場合)
        # target_model が lowload でない、または system prompt があれば追加
        if target_model != self.lowload_model_name and self.system_prompt:
             messages.append(ChatMessage(role="system", content=self.system_prompt))

        # 2. Deep Cache Summary (履歴の最初に追加)
        if deep_cache_summary:
            # System Prompt がない場合は User として追加
            if not messages or messages[0].role != "system":
                messages.append(ChatMessage(role="user", content=f"【長期記憶からの参考情報】\n{deep_cache_summary}"))
            else:
                # System Prompt がある場合はそれに追加するか、別メッセージにする
                # ここではSystem Promptに追記してみる（長さに注意）
                 messages[0].content += f"\n\n【長期記憶からの参考情報】\n{deep_cache_summary}"
                 print("Deep Cache information appended to system prompt.")


        # 3. Chat History
        if chat_history:
            for entry in chat_history:
                role = entry.get("role")
                parts = entry.get("parts", [])
                if not role or not parts: continue

                # Mistral の role に変換 ('model' -> 'assistant')
                mistral_role = "assistant" if role == "model" else "user"

                # Lowload モデルで system が使えない場合の代替処理
                if target_model == self.lowload_model_name and mistral_role not in self.SUPPORTED_LOWLOAD_ROLES:
                    print(f"Warning: Skipping unsupported role '{mistral_role}' for lowload model {self.lowload_model_name}")
                    continue

                # parts を結合して content を作成 (画像は履歴に含まれない想定)
                text_content = " ".join(p.get("text", "") for p in parts if "text" in p)
                if text_content:
                     messages.append(ChatMessage(role=mistral_role, content=text_content))

        # 4. Current User Input (content_parts)
        current_text_parts = []
        current_image_parts = []
        for part in content_parts:
             if "text" in part:
                 current_text_parts.append(part["text"])
             elif "inline_data" in part:
                 # 画像データをBase64エンコードして保持
                 try:
                     mime_type = part["inline_data"]["mime_type"]
                     data = part["inline_data"]["data"]
                     if isinstance(data, bytes):
                         b64_data = base64.b64encode(data).decode('utf-8')
                         # Mistral (Pixtral) は URL 形式で渡す必要がある
                         image_url = f"data:{mime_type};base64,{b64_data}"
                         current_image_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                     else:
                          print("Warning: Invalid image data type in content_parts, skipping.")
                 except Exception as e:
                     print(f"Warning: Failed to process image data: {e}, skipping.")

        # Mistral API (v3) 形式の content リストを作成
        mistral_content = []
        if current_text_parts:
            mistral_content.append({"type": "text", "text": "\n".join(current_text_parts)})
        mistral_content.extend(current_image_parts)

        if mistral_content:
             messages.append(ChatMessage(role="user", content=mistral_content)) # type: ignore (リスト形式のcontentを渡す)

        return messages

    async def _call_mistral_api(
        self,
        model_name: str,
        messages: List[ChatMessage]
        ) -> Tuple[str, str]:
        """Mistral API を呼び出し、結果またはエラーメッセージを返す"""
        if not self.client:
            return model_name, self.format_error_message(ERROR_TYPE_INTERNAL, "Mistral client not initialized.")
        if not model_name:
             return "No Model", self.format_error_message(ERROR_TYPE_INTERNAL, "Mistral model name not specified.")

        try:
            print(f"Calling Mistral API ({model_name})... Messages: {len(messages)}")
            # 非同期呼び出し (SDKが非同期をサポートしているか確認 -> していない場合は asyncio.to_thread)
            # response = await self.client.chat(model=model_name, messages=messages) # 同期の場合
            response = await asyncio.to_thread(
                self.client.chat,
                model=model_name,
                messages=messages
                # safe_prompt=True # 必要なら安全フィルター有効化
            )
            print(f"Mistral API ({model_name}) response received.")

            if response.choices and response.choices[0].message:
                response_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if response_content is None: response_content = "" # Noneの場合空文字列に

                if finish_reason == "length":
                    # 最大トークン数に達した場合
                    print(f"Warning: Mistral response truncated due to maximum length ({model_name}).")
                    error_type, error_detail = ERROR_TYPE_INVALID_ARGUMENT, "Output exceeded maximum token limit."
                    # 途中までのテキストとエラーメッセージを結合
                    return model_name, response_content + f"\n\n...{self.format_error_message(error_type, error_detail)}"
                elif finish_reason == "stop":
                     # 正常終了
                     return model_name, response_content
                elif finish_reason == "tool_calls":
                     # TODO: ツール呼び出し対応が必要な場合はここに実装
                     print(f"Warning: Tool call requested by Mistral model ({model_name}), but not handled.")
                     return model_name, response_content + "\n\n(Tool call detected but not processed)" # 一旦テキストを返す
                else: # 'error' や不明な理由
                     print(f"Warning: Mistral response finished with reason: {finish_reason} ({model_name}).")
                     return model_name, self.format_error_message(ERROR_TYPE_UNKNOWN, f"Finished with reason: {finish_reason}")
            else:
                print(f"Warning: No choices or message content in Mistral response ({model_name}).")
                return model_name, self.format_error_message(ERROR_TYPE_UNKNOWN, "Empty response from API.")

        except Exception as e:
            print(f"Error during Mistral API call ({model_name}): {e}")
            error_type, error_detail = _map_mistral_exception_to_error_type(e)
            # generate_response でキャッチしてフォールバックするために例外を再送出
            raise e


    async def generate_response(
        self,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None,
    ) -> Tuple[str, str]:

        # --- Pixtral (画像対応) モデルかどうかの簡易チェック ---
        # content_parts に画像が含まれているか確認
        has_images = any("inline_data" in part for part in content_parts)
        target_model_name = self.primary_model_name

        # 画像があるのにモデル名に 'pixtral' が含まれていない場合、警告またはエラー
        if has_images and target_model_name and "pixtral" not in target_model_name.lower():
            # ここでは警告を出し、テキストのみで処理を試みるか、エラーを返すか選択
            print(f"Warning: Images provided but the primary model '{target_model_name}' might not support vision. Attempting anyway or consider using a vision model like Pixtral.")
            # エラーにする場合:
            # return target_model_name, self.format_error_message(ERROR_TYPE_UNSUPPORTED_FEATURE, f"Model {target_model_name} does not support image input.")

        # 1. Mistral形式に変換
        try:
            mistral_messages = self._convert_history_to_mistral_chat(
                content_parts, chat_history, deep_cache_summary, target_model_name
            )
            if not mistral_messages: # 特に最後のユーザー入力が空だった場合など
                 return target_model_name or "No Model", self.format_error_message(ERROR_TYPE_INVALID_ARGUMENT, "No content to send.")
        except Exception as e:
            print(f"Error converting history/parts to Mistral format: {e}")
            return target_model_name or "No Model", self.format_error_message(ERROR_TYPE_INTERNAL, f"Failed to format request: {e}")

        # 2. プライマリモデルで試行
        if self.primary_model_name:
            current_model_to_try = self.primary_model_name
            try:
                return await self._call_mistral_api(current_model_to_try, mistral_messages)
            except Exception as e:
                error_type, error_detail = _map_mistral_exception_to_error_type(e)
                # Mistralではレートリミット以外のエラーでのフォールバックはあまり意味がないかもしれない
                # レートリミットの場合のみフォールバックを試みる
                if self.is_rate_limit_error(e) and self.secondary_model_name and self.secondary_model_name != self.primary_model_name:
                    print(f"Mistral Primary model ({self.primary_model_name}) rate limited. Falling back to secondary ({self.secondary_model_name}).")
                    # フォールバック処理へ
                else:
                    print(f"Error with Mistral Primary model ({self.primary_model_name}): {e}. No fallback possible or error is not rate limit.")
                    return self.primary_model_name, self.format_error_message(error_type, error_detail)
        else:
             print("Error: Mistral Primary model name not set.")
             # フォールバックを試みる
             pass

        # 3. セカンダリモデルで試行 (プライマリ失敗時)
        if self.secondary_model_name:
             current_model_to_try = self.secondary_model_name
             print(f"Attempting fallback with Mistral Secondary model ({current_model_to_try})...")
             # セカンダリモデル用にメッセージを再変換する必要があるか？ (通常は不要)
             # ただし、セカンダリが画像非対応の場合は画像を除去する必要がある
             secondary_messages = mistral_messages
             secondary_has_images = has_images
             if secondary_has_images and "pixtral" not in current_model_to_try.lower():
                 print(f"Warning: Secondary model '{current_model_to_try}' might not support images. Removing images from request.")
                 # 最後のユーザーメッセージから画像を除去する処理
                 if secondary_messages and secondary_messages[-1].role == "user" and isinstance(secondary_messages[-1].content, list):
                     secondary_messages[-1].content = [part for part in secondary_messages[-1].content if part["type"] == "text"] # type: ignore
                     if not secondary_messages[-1].content: # テキストもなかったらメッセージ自体を削除？
                         print("Warning: User message became empty after removing images for secondary model.")
                         # ここではメッセージリストはそのままにしてAPIに投げてみる

             try:
                 return await self._call_mistral_api(current_model_to_try, secondary_messages)
             except Exception as e:
                 print(f"Error with Mistral Secondary model ({current_model_to_try}): {e}")
                 error_type, error_detail = _map_mistral_exception_to_error_type(e)
                 return current_model_to_try, self.format_error_message(error_type, error_detail)
        else:
            print("Error: No secondary Mistral model available for fallback.")
            # プライマリのエラーメッセージを返す (上で既に返されているはずだが念のため)
            primary_error_msg = self.format_error_message(ERROR_TYPE_INTERNAL, "Primary model failed and no secondary model available.")
            return self.primary_model_name or "No Model", primary_error_msg


    async def generate_lowload_response(self, prompt: str) -> Optional[str]:
        if not self.client or not self.lowload_model_name:
            print("Warning: Mistral Lowload model/client is not available.")
            return None

        # Lowload モデル用のメッセージリスト作成 (System Promptなし)
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            # 低負荷モデルはフォールバックしない
            model_name, response_text = await self._call_mistral_api(self.lowload_model_name, messages)
            # エラーメッセージが返ってきた場合は None を返す
            if self._is_error_message(response_text):
                 print(f"Lowload response generated an error message: {response_text}")
                 return None
            return response_text
        except Exception as e:
             # _call_mistral_api 内でログは出力されるはず
             print(f"Lowload API call failed for model {self.lowload_model_name}.")
             return None # エラー時はNone


    def format_error_message(self, error_type: str, detail: Optional[str] = None) -> str:
        # bot_constants からプラナ風エラーメッセージを取得
        if error_type == ERROR_TYPE_RATE_LIMIT:
            return bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED # Gemini用を流用
        elif error_type == ERROR_TYPE_INVALID_ARGUMENT:
            if detail and "Input too large" in detail:
                 return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG + " (入力過大)"
            elif detail and "token limit" in detail:
                 return bot_constants.ERROR_MSG_MAX_TEXT_SIZE
            elif detail and "Invalid image data" in detail:
                 return bot_constants.ERROR_MSG_IMAGE_READ_FAIL + " (データ破損)"
            return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG # Gemini用を流用
        elif error_type == ERROR_TYPE_BLOCKED_PROMPT: # Mistralでは明示されない可能性
            return bot_constants.ERROR_MSG_GEMINI_BLOCKED_PROMPT
        elif error_type == ERROR_TYPE_BLOCKED_RESPONSE: # Mistralでは明示されない可能性
            return bot_constants.ERROR_MSG_GEMINI_BLOCKED_RESPONSE
        elif error_type == ERROR_TYPE_API_ERROR:
             if detail and "Authentication failed" in detail:
                  return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (認証失敗)"
             if detail and "Connection error" in detail:
                  return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (接続失敗)"
             if detail and "server error" in detail:
                  return bot_constants.ERROR_MSG_GEMINI_API_ERROR + " (サーバーエラー)"
             return bot_constants.ERROR_MSG_GEMINI_API_ERROR
        elif error_type == ERROR_TYPE_INTERNAL:
            return bot_constants.ERROR_MSG_INTERNAL
        elif error_type == ERROR_TYPE_UNSUPPORTED_FEATURE:
             return bot_constants.ERROR_MSG_INTERNAL + " (未対応機能)"
        else: # ERROR_TYPE_UNKNOWN
            return bot_constants.ERROR_MSG_GEMINI_UNKNOWN

    def is_rate_limit_error(self, exception: Exception) -> bool:
        return isinstance(exception, MistralAPIException) and exception.status_code == 429

    def is_invalid_argument_error(self, exception: Exception) -> bool:
         # 400 Bad Request or 422 Unprocessable Entity
        return isinstance(exception, MistralAPIException) and exception.status_code in [400, 422]