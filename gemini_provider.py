# gemini_provider.py
# (Gemini API用プロバイダーの実装 - google.genai SDK 使用)

# --- ライブラリインポート ---
import google.genai as genai
from google.genai import types
from google.genai import errors as genai_errors # エラーモジュールをインポート
import asyncio
import base64
import re
import httpx # APIErrorからのステータスコード判断のために追加
from typing import List, Dict, Any, Optional, Tuple, Literal

# --- 内部モジュールインポート ---
from llm_provider import (
    LLMProvider, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_INVALID_ARGUMENT,
    ERROR_TYPE_BLOCKED_PROMPT, ERROR_TYPE_BLOCKED_RESPONSE,
    ERROR_TYPE_API_ERROR, ERROR_TYPE_UNKNOWN, ERROR_TYPE_INTERNAL,
    ERROR_TYPE_UNSUPPORTED_FEATURE
)
import bot_constants

# --- Gemini エラーマッピング (google.genai 用) ---

def _map_gemini_error_to_error_type(e: Exception) -> Tuple[str, Optional[str]]:
    """Google GenAI SDKの例外を共通エラータイプと詳細にマッピング (APIError中心)"""
    detail = str(e)
    if isinstance(e, genai_errors.APIError):
        message = detail.lower()
        # HTTPStatusErrorの情報を活用 (APIErrorがhttpx.HTTPStatusErrorをラップしている可能性がある)
        status_code = getattr(e, 'status_code', None) # APIErrorオブジェクトに直接status_codeがあるか試す
        if status_code is None and hasattr(e, 'response') and hasattr(e.response, 'status_code'):
             status_code = e.response.status_code # レスポンスオブジェクト経由で取得

        # 499 CANCELLED もここで拾う
        if status_code == 499:
             return ERROR_TYPE_API_ERROR, f"Request cancelled: {detail}"

        if status_code == 400 or "invalid" in message or "bad request" in message:
            # 400 Bad Request または invalid/bad request キーワード
            if "unsupported mime type" in message:
                 match = re.search(r"unsupported mime type: (.*?)\.", message)
                 mime_type_error = match.group(1) if match else "不明"
                 return ERROR_TYPE_INVALID_ARGUMENT, f"Unsupported MIME type found in request ({mime_type_error})."
            elif "prompt is too long" in message or "request payload size" in message or "context length" in message:
                 return ERROR_TYPE_INVALID_ARGUMENT, "Input too large (text or image)."
            elif "invalid base64" in message:
                 return ERROR_TYPE_INVALID_ARGUMENT, "Invalid image data."
            return ERROR_TYPE_INVALID_ARGUMENT, detail
        elif status_code == 401 or "api key not valid" in message or "authentication" in message:
             return ERROR_TYPE_API_ERROR, "Authentication failed (Invalid API Key)."
        elif status_code == 403 or "permission denied" in message:
             return ERROR_TYPE_API_ERROR, "Permission denied."
        elif status_code == 429 or "quota" in message or "rate limit" in message:
            return ERROR_TYPE_RATE_LIMIT, detail
        elif status_code >= 500 or "internal server error" in message:
             return ERROR_TYPE_API_ERROR, f"API server error ({status_code or 'N/A'}): {detail}"
        elif status_code is not None:
            return ERROR_TYPE_API_ERROR, f"API error ({status_code}): {detail}"
        else: # ステータスコード不明なAPIエラー
            return ERROR_TYPE_API_ERROR, detail
    elif isinstance(e, httpx.ConnectError): # httpx由来の接続エラー
         return ERROR_TYPE_API_ERROR, f"Connection error: {detail}"
    elif isinstance(e, asyncio.TimeoutError): # 明示的なタイムアウトエラー
        return ERROR_TYPE_API_ERROR, f"API request timed out: {detail}" # より具体的なエラータイプが必要なら追加 (例: ERROR_TYPE_TIMEOUT)

    # 他の特定のSDK例外クラスがあればここで追加 (現在はAPIError中心)
    # elif isinstance(e, genai_errors.BlockedPromptError): # もしあれば
    #     return ERROR_TYPE_BLOCKED_PROMPT, "Prompt blocked by safety filter."

    else: # SDKのエラー以外または不明なAPIエラー
        return ERROR_TYPE_UNKNOWN, detail


def _map_gemini_finish_reason_to_error(finish_reason: types.FinishReason, response: Any) -> Optional[Tuple[str, Optional[str]]]:
    """Geminiの応答のfinish_reasonを共通エラータイプと詳細にマッピング (types.FinishReasonを使用)"""
    if finish_reason == types.FinishReason.SAFETY:
        block_detail = "Safety settings triggered."
        if hasattr(response, 'candidates') and response.candidates:
             candidate = response.candidates[0]
             # safety_ratings が types.SafetyRating のリストであることを期待
             if hasattr(candidate, 'safety_ratings') and isinstance(candidate.safety_ratings, list):
                 blocked_categories = [
                     r.category.name for r in candidate.safety_ratings
                     if hasattr(r, 'probability') and hasattr(r.probability, 'name') and r.probability.name not in ["NEGLIGIBLE", "LOW"]
                 ]
                 if blocked_categories:
                     block_detail += f" Categories: {', '.join(blocked_categories)}"
        return ERROR_TYPE_BLOCKED_RESPONSE, block_detail
    elif finish_reason == types.FinishReason.MAX_TOKENS:
        return ERROR_TYPE_INVALID_ARGUMENT, "Output exceeded maximum token limit."
    elif finish_reason in [types.FinishReason.RECITATION, types.FinishReason.OTHER]:
        return ERROR_TYPE_UNKNOWN, f"Stopped due to reason: {finish_reason.name}"
    return None # FINISH_REASON_UNSPECIFIED や STOP はエラーではない

def _map_gemini_prompt_feedback_to_error(response: Any) -> Optional[Tuple[str, Optional[str]]]:
    """Geminiのprompt_feedbackを共通エラータイプと詳細にマッピング"""
    if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
        # block_reason は types.BlockReason (Enum) であることを期待
        reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
        return ERROR_TYPE_BLOCKED_PROMPT, f"Prompt blocked due to reason: {reason}"
    return None

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.client: Optional[genai.Client] = None # Clientオブジェクトを保持
        self.primary_model_name: Optional[str] = None
        self.secondary_model_name: Optional[str] = None
        self.lowload_model_name: Optional[str] = None
        self.system_prompt: str = ""
        # Safety Settings は API 呼び出し時に渡さないように変更
        # self.safety_settings = [...]
        # Generation Config も API 呼び出し時に渡さないように変更
        # self.generation_config = types.GenerationConfig(...)

    async def initialize(self, api_key: str, model_config: Dict[str, str], system_prompt: str, base_url: Optional[str] = None) -> bool:
        """GeminiProviderの初期化 (google.genai SDK)"""
        try:
            # Client を初期化
            # google.genai.Client はデフォルトで同期/非同期の両方を扱うための構造を持っている
            self.client = genai.Client(api_key=api_key)
            self.system_prompt = system_prompt

            # モデル名を保存
            self.primary_model_name = model_config.get('primary')
            self.secondary_model_name = model_config.get('secondary')
            self.lowload_model_name = model_config.get('lowload')

            # モデル名の検証 (APIコールはしない)
            if not self.primary_model_name and not self.secondary_model_name:
                print("Error: No primary or secondary Gemini model name configured.")
                return False
            if not self.lowload_model_name:
                print("Warning: Gemini Lowload model name unavailable. Related features might be limited.")

            print(f"Gemini provider initialized with Client. Models: P={self.primary_model_name}, S={self.secondary_model_name}, L={self.lowload_model_name}")
            return True
        except Exception as e:
            print(f"Exception within GeminiProvider.initialize: {e}")
            self.client = None
            return False

    def _prepare_gemini_contents(
        self,
        content_parts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        deep_cache_summary: Optional[str] = None,
        include_system_prompt: bool = True # システムプロンプトをコンテンツに含めるか (Lowload用)
    ) -> List[types.Content]:
        """履歴、Deep Cache、現在の入力を google.genai の contents 形式に変換"""
        gemini_contents: List[types.Content] = []

        # 1. 履歴コンテンツを生成
        history_contents_only: List[types.Content] = []
        if chat_history:
             for entry in chat_history:
                  role = entry.get("role")
                  parts = entry.get("parts", [])
                  if not role or not parts: continue
                  gemini_parts: List[types.Part] = []
                  for part in parts:
                       if "text" in part: gemini_parts.append(types.Part.from_text(text=part["text"]))
                       elif "inline_data" in part:
                            try:
                                 mime_type = part["inline_data"]["mime_type"]
                                 data = part["inline_data"]["data"]
                                 if isinstance(data, bytes): gemini_parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type, data=data)))
                            except Exception as e: print(f"Warning: Failed to process image data from history: {e}, skipping image.")
                  if gemini_parts: history_contents_only.append(types.Content(role=role, parts=gemini_parts))


        # 2. システムプロンプト、Deep Cache、現在の入力をまとめた最後のユーザーコンテンツを生成
        last_user_parts: List[types.Part] = []
        last_user_text = ""

        # システムプロンプトを含める場合
        if include_system_prompt and self.system_prompt:
             last_user_text += self.system_prompt

        # Deep Cacheを含める場合
        if deep_cache_summary:
             if last_user_text: last_user_text += "\n\n"
             last_user_text += f"【長期記憶からの参考情報】\n{deep_cache_summary}"

        # 現在の入力のテキストパート
        current_text_parts = [p["text"] for p in content_parts if "text" in p]
        if current_text_parts:
             if last_user_text: last_user_text += "\n\n"
             last_user_text += "\n".join(current_text_parts)

        # まとめたテキストをパートに追加
        if last_user_text:
             last_user_parts.append(types.Part.from_text(text=last_user_text))

        # 現在の入力の画像パートを最後のユーザーメッセージに追加
        current_image_parts: List[types.Part] = []
        for part in content_parts:
             if "inline_data" in part:
                 try:
                     mime_type = part["inline_data"]["mime_type"]
                     data = part["inline_data"]["data"]
                     if isinstance(data, bytes):
                           blob = types.Blob(mime_type=mime_type, data=data)
                           current_image_parts.append(types.Part(inline_data=blob))
                     else:
                          print(f"Warning: Invalid image data type in current input ({type(data)}), skipping image.")
                 except Exception as e:
                      print(f"Warning: Failed to process image data from current input: {e}, skipping image.")

        last_user_parts.extend(current_image_parts)

        # 3. 最終的な Contents リストは 履歴 + 最後のユーザーコンテンツ
        gemini_contents = history_contents_only
        if last_user_parts:
             # 履歴の最後のロールを確認し、必要であればダミー応答を挿入 (現在は行わない方針だが、コード構造は維持)
             # if history_contents_only and history_contents_only[-1].role == 'user':
             #      print("Warning: Inserting dummy model response between consecutive user messages.")
             #      gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text="...")])) # ダミー応答
             gemini_contents.append(types.Content(role="user", parts=last_user_parts))


        # 履歴も現在の入力もどちらも空の場合
        if not gemini_contents:
             print("Warning: No content to send to Gemini API after preparation.")
             return []

        return gemini_contents


    async def _generate_content_internal(
        self,
        model_name: str,
        content_parts: List[Dict[str, Any]], # 現在のユーザー入力パーツ (テキストと画像)
        chat_history: Optional[List[Dict[str, Any]]] = None, # 過去の会話履歴リスト [{'role': ..., 'parts': [...]}]
        deep_cache_summary: Optional[str] = None, # Deep Cacheサマリーテキスト
        include_system_prompt_in_content: bool = True # システムプロンプトをコンテンツに含めるか (Lowload用)
        # timeout: Optional[float] = None # <-- timeout パラメータを削除
    ) -> Tuple[str, str]:
        """Gemini API呼び出しのコアロジック (google.genai SDK)"""
        if not self.client: return model_name, self.format_error_message(ERROR_TYPE_INTERNAL, "Client not initialized.")
        if not model_name: return "No Model", self.format_error_message(ERROR_TYPE_INTERNAL, "Model name not specified.")
        # *** self.client.aio が存在することを確認 ***
        if not hasattr(self.client, 'aio') or not hasattr(self.client.aio, 'models'):
             print("CRITICAL: Gemini client.aio.models not available.")
             return model_name, self.format_error_message(ERROR_TYPE_INTERNAL, "Async models interface not available.")


        # 1. contents リストを準備
        try:
            gemini_contents = self._prepare_gemini_contents(
                 content_parts=content_parts,
                 chat_history=chat_history,
                 deep_cache_summary=deep_cache_summary,
                 include_system_prompt=include_system_prompt_in_content # フラグを渡す
             )
            if not gemini_contents:
                 return model_name, self.format_error_message(ERROR_TYPE_INVALID_ARGUMENT, "No content to send.")
        except Exception as e:
            print(f"Error preparing contents for Gemini API: {e}")
            import traceback
            traceback.print_exc()
            return model_name, self.format_error_message(ERROR_TYPE_INTERNAL, f"Content preparation failed: {e}")


        # 2. API 呼び出し (非同期)
        try:
            print(f"Calling Gemini API ({model_name}) with {len(gemini_contents)} content blocks...")
            # *** 修正箇所 ***
            # 非同期メソッドは self.client.aio.models にあると想定して呼び出す
            response = await self.client.aio.models.generate_content( # <-- ここは前回の修正でOK
                model=model_name,
                contents=gemini_contents,
                # generation_config は削除済み
                # safety_settings は削除済み
                # stream は削除済み (generate_content は非ストリーミング用)
                # timeout は generate_content の引数ではないため削除
            )
            print(f"Gemini API ({model_name}) response received.")

            # 3. 応答のパースとエラーチェック
            response_text = None
            # finish_reason は response.candidates[0].finish_reason に格納されるはず
            finish_reason: types.FinishReason = types.FinishReason.FINISH_REASON_UNSPECIFIED # 初期値
            error_type = None
            error_detail = None

            # Prompt Feedback チェック (ブロックされたか)
            prompt_error = _map_gemini_prompt_feedback_to_error(response)
            if prompt_error:
                error_type, error_detail = prompt_error
                print(f"Warning: Prompt blocked by Gemini ({model_name}). Reason: {error_detail}")
                return model_name, self.format_error_message(error_type, error_detail)

            # Candidates チェック (正常応答または他の終了理由)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0];
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason is not None:
                    finish_reason = candidate.finish_reason # types.FinishReason Enumのはず

                # テキスト抽出
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

                # Finish Reason によるエラーチェック
                # FINISH_REASON_UNSPECIFIED や STOP 以外はエラーの可能性あり
                if finish_reason != types.FinishReason.FINISH_REASON_UNSPECIFIED and finish_reason != types.FinishReason.STOP:
                     finish_error = _map_gemini_finish_reason_to_error(finish_reason, response)
                     if finish_error:
                         error_type, error_detail = finish_error
                         print(f"Warning: Response generation stopped by Gemini ({model_name}). Reason: {finish_reason.name}, Detail: {error_detail}")
                         # トークン制限で停止した場合、途中までのテキストを返す
                         if error_type == ERROR_TYPE_INVALID_ARGUMENT and "token limit" in (error_detail or "") and response_text:
                             return model_name, response_text + f"\n\n...{self.format_error_message(error_type, error_detail)}"
                         # SAFETY ブロックの場合、応答テキストがないことが多いが、念のためチェック
                         if error_type == ERROR_TYPE_BLOCKED_RESPONSE and response_text:
                              return model_name, response_text + f"\n\n...{self.format_error_message(error_type, error_detail)}"
                         return model_name, self.format_error_message(error_type, error_detail)
            # 古い形式？ (念のため - generate_content_stream で使う可能性もあるがここでは generate_content のみ)
            # elif hasattr(response, 'text'):
            #      response_text = response.text
            #      finish_reason = types.FinishReason.STOP # 成功とみなす


            # 4. 結果を返す
            if response_text is not None:
                return model_name, response_text
            else:
                # テキストがない場合 (候補がない、または空の応答)
                finish_reason_name = finish_reason.name if finish_reason else "UNKNOWN"
                print(f"Warning: No text content in Gemini response ({model_name}). Finish Reason: {finish_reason_name}")
                # finish_reason が SAFETY なら上で処理されているはずだが、念のため
                if finish_reason == types.FinishReason.SAFETY:
                     return model_name, self.format_error_message(ERROR_TYPE_BLOCKED_RESPONSE, "Response likely blocked by safety filter (no text).")
                return model_name, self.format_error_message(ERROR_TYPE_UNKNOWN, f"No text content received. Finish Reason: {finish_reason_name}")

        except Exception as e:
            print(f"Error during Gemini API call ({model_name}): {e}")
            # 共通エラーマッパーを使用
            error_type, error_detail = _map_gemini_error_to_error_type(e)
            return model_name, self.format_error_message(error_type, error_detail)


    async def generate_response(
        self,
        content_parts: List[Dict[str, Any]], # 現在のユーザー入力パーツ
        chat_history: Optional[List[Dict[str, Any]]] = None, # 過去の会話履歴リスト [{'role': ..., 'parts': ...]}]
        deep_cache_summary: Optional[str] = None, # Deep Cacheサマリーテキスト
    ) -> Tuple[str, str]:
        # Primary model で試行
        if self.primary_model_name:
            try:
                # generate_response からは timeout を渡さない
                model_name, response_text = await self._generate_content_internal(
                    model_name=self.primary_model_name,
                    content_parts=content_parts,
                    chat_history=chat_history,
                    deep_cache_summary=deep_cache_summary,
                    include_system_prompt_in_content=True, # 通常応答はシステムプロンプトを含む
                    # timeout 引数は削除
                )
                # エラー応答でないか確認 (_is_error_messageを使用)
                if not self._is_error_message(response_text):
                    return model_name, response_text # 成功

                # エラーメッセージの場合、レートリミットか判定しフォールバックを試みる
                # format_error_message が返す文字列に特定のキーワードが含まれるか判定
                if bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED in response_text: # レートリミットメッセージと一致するか
                     print(f"Gemini Primary model ({self.primary_model_name}) returned rate limit error message. Trying secondary.")
                     pass # フォールバック処理へ
                else:
                     # レートリミット以外のエラーメッセージはそのまま返す
                     print(f"Gemini Primary model ({self.primary_model_name}) returned non-rate limit error message. Not falling back.")
                     return model_name, response_text

            except Exception as e: # _generate_content_internal が例外を投げた場合
                 print(f"Exception during primary model call ({self.primary_model_name}): {e}")
                 error_type, error_detail = _map_gemini_error_to_error_type(e)
                 if not self.is_rate_limit_error(e): # 例外自体がレートリミットか判定
                     # レートリミット以外の例外ならここでエラーを返す
                     return self.primary_model_name or "No Model", self.format_error_message(error_type, error_detail)
                 # レートリミット例外ならフォールバックへ
                 print(f"Gemini Primary model ({self.primary_model_name}) exception indicates rate limit. Trying secondary.")
                 pass
        else:
             print("Gemini Primary model name not available. Attempting Secondary model.")

        # Secondary model で試行 (Primary がない or レートリミットの場合)
        if self.secondary_model_name:
            print(f"Falling back to secondary model: {self.secondary_model_name}")
            try:
                # セカンダリモデルで実行
                 # generate_response からは timeout を渡さない
                 model_name, response_text = await self._generate_content_internal(
                    model_name=self.secondary_model_name,
                    content_parts=content_parts,
                    chat_history=chat_history,
                    deep_cache_summary=deep_cache_summary,
                    include_system_prompt_in_content=True # 通常応答はシステムプロンプトを含む
                    # timeout 引数は削除
                )
                 return model_name, response_text # セカンダリの結果をそのまま返す (エラー含む)
            except Exception as e:
                print(f"Error with Gemini Secondary model ({self.secondary_model_name}): {e}")
                error_type, error_detail = _map_gemini_error_to_error_type(e)
                return self.secondary_model_name, self.format_error_message(error_type, error_detail)
        else:
            print("Error: Both primary and secondary Gemini models are unavailable or failed.")
            # プライマリのエラーか、モデルがない旨のエラーを返す
            # プライマリがレートリミットだったか不明なため、汎用的なエラーメッセージを返す
            return self.primary_model_name or "No Model", self.format_error_message(ERROR_TYPE_API_ERROR, "Primary model failed and no secondary model available.")


    # generate_lowload_response の引数から timeout を削除 (search_handler で設定する)
    async def generate_lowload_response(self, prompt: str) -> Optional[str]:
        """低負荷モデルでシンプルな応答を生成 (google.genai SDK)"""
        if not self.client:
            print("Warning: Gemini client not initialized for lowload response.")
            return None
        if not self.lowload_model_name:
            print("Warning: Gemini Lowload model name is not available.")
            return None

        # *** self.client.aio が存在することを確認 ***
        if not hasattr(self.client, 'aio') or not hasattr(self.client.aio, 'models'):
             print("CRITICAL: Gemini client.aio.models not available for lowload.")
             return None # Lowloadではエラーメッセージは返さない


        try:
            # Lowload はシステムプロンプトなし、履歴なしで呼び出す
            # _generate_content_internal に timeout は渡さない
            model_name, response_text = await self._generate_content_internal(
                model_name=self.lowload_model_name,
                content_parts=[{'text': prompt}], # 単純なテキストをパーツとして渡す
                chat_history=None,
                deep_cache_summary=None, # Deep CacheはLowloadの用途によって必要/不要が変わるが、ここでは含めない方針
                include_system_prompt_in_content=False, # Lowloadではシステムプロンプト不要
                # timeout 引数は削除
            )
            # エラーメッセージが返ってきた場合は None を返す (_is_error_messageを使用)
            if self._is_error_message(response_text):
                 print(f"Lowload response generated an error: {response_text}")
                 return None
            return response_text.strip() # 正常ならテキストを返す
        except Exception as e:
             # _generate_content_internal 内でエラー処理されているはずだが、念のため
             print(f"Error during Gemini Lowload API call ({self.lowload_model_name}): {e}")
             return None

    def format_error_message(self, error_type: str, detail: Optional[str] = None) -> str:
        # エラータイプに応じて bot_constants のメッセージを返す (既存ロジックを維持)
        if error_type == ERROR_TYPE_RATE_LIMIT: return bot_constants.ERROR_MSG_GEMINI_RESOURCE_EXHAUSTED
        elif error_type == ERROR_TYPE_INVALID_ARGUMENT:
            # detail からもう少し情報を付加
            if detail and "Unsupported MIME type" in detail:
                match = re.search(r"Unsupported MIME type.*?\((.*?)\)", detail)
                mime_type = match.group(1) if match else "不明"
                return bot_constants.ERROR_MSG_ATTACHMENT_UNSUPPORTED + f" ({mime_type})"
            elif detail and ("Input too large" in detail or "token limit" in detail or "context length" in detail or "request payload size" in detail):
                # 入力過大と出力過大、context length エラーをまとめる
                return bot_constants.ERROR_MSG_MAX_TEXT_SIZE # より汎用的なメッセージに変更
            elif detail and "Invalid image data" in detail:
                 return bot_constants.ERROR_MSG_IMAGE_READ_FAIL + " (データ破損)"
            return bot_constants.ERROR_MSG_GEMINI_INVALID_ARG
        elif error_type == ERROR_TYPE_BLOCKED_PROMPT: return bot_constants.ERROR_MSG_GEMINI_BLOCKED_PROMPT
        elif error_type == ERROR_TYPE_BLOCKED_RESPONSE: return bot_constants.ERROR_MSG_GEMINI_BLOCKED_RESPONSE
        elif error_type == ERROR_TYPE_API_ERROR:
             prefix = bot_constants.ERROR_MSG_GEMINI_API_ERROR
             if detail:
                 detail_lower = detail.lower()
                 if "authentication failed" in detail_lower or "api key not valid" in detail_lower: prefix += " (認証失敗)"
                 elif "permission denied" in detail_lower: prefix += " (権限不足)"
                 elif "connection error" in detail_lower or "connecterror" in detail_lower: prefix += " (接続失敗)"
                 elif "server error" in detail_lower or "internal server error" in detail_lower: prefix += " (サーバーエラー)"
                 # 499 CANCELLED や timeout に関連するエラーメッセージを拾う
                 elif "cancelled" in detail_lower or "timeout" in detail_lower or "499" in detail_lower:
                      prefix += " (タイムアウト/キャンセル)" # タイムアウトやキャンセル関連と推測
                 # safety_settings に関連するエラーメッセージを拾う
                 elif "safety_settings" in detail_lower or "harmcategory" in detail_lower or "blockthreshold" in detail_lower:
                      prefix += " (安全設定エラー)" # 安全設定関連のエラーと推測

                 else: prefix += f" ({detail[:50]}...)" # 不明なAPIエラー詳細
             return prefix
        elif error_type == ERROR_TYPE_INTERNAL: return bot_constants.ERROR_MSG_INTERNAL
        elif error_type == ERROR_TYPE_UNKNOWN:
            prefix = bot_constants.ERROR_MSG_GEMINI_UNKNOWN
            if detail: prefix += f" ({detail[:50]}...)"
            return prefix
        elif error_type == ERROR_TYPE_UNSUPPORTED_FEATURE: return bot_constants.ERROR_MSG_INTERNAL + " (未対応機能)"
        else: return bot_constants.ERROR_MSG_GEMINI_UNKNOWN

    def is_rate_limit_error(self, exception: Exception) -> bool:
        """例外オブジェクトがレートリミットエラーか判定"""
        # APIErrorの場合、メッセージ内容やステータスコードで判定
        if isinstance(exception, genai_errors.APIError):
            message = str(exception).lower()
            status_code = getattr(exception, 'status_code', None)
            if status_code is None and hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
                 status_code = exception.response.status_code
            return status_code == 429 or "quota" in message or "rate limit" in message
        # 他にレートリミットに相当する特定の例外クラスがあれば追加
        # return isinstance(exception, genai_errors.ResourceLimitError) or ...
        return False # それ以外の例外はレートリミットではないと判定

    def is_invalid_argument_error(self, exception: Exception) -> bool:
        """例外オブジェクトが無効な引数エラーか判定"""
        # APIErrorの場合、メッセージ内容やステータスコードで判定
        if isinstance(exception, genai_errors.APIError):
            message = str(exception).lower()
            status_code = getattr(exception, 'status_code', None)
            if status_code is None and hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
                 status_code = exception.response.status_code
            # safety_settings が原因の Invalid Argument も拾う
            return status_code == 400 or "invalid" in message or "bad request" in message or "unsupported" in message or "prompt is too long" in message or "request payload size" in message or "context length" in message or "safety_settings" in message or "harmcategory" in message or "blockthreshold" in message
        # 他に無効な引数に相当する特定の例外クラスがあれば追加
        # return isinstance(exception, genai_errors.InvalidArgumentError) or ...
        return False # それ以外の例外は無効な引数ではないと判定


    def get_model_name(self, model_type: Literal["primary", "secondary", "lowload"]) -> Optional[str]:
        """保持しているモデル名文字列を返す"""
        if model_type == "primary": return self.primary_model_name
        elif model_type == "secondary": return self.secondary_model_name
        elif model_type == "lowload": return self.lowload_model_name
        return None

# search_handler.py の generate_dsrc_report 関数内の修正は
# generate_lowload_response 呼び出しに timeout を追加する形で行われました。
# gemini_provider.py 側では _generate_content_internal が timeout を受け取らなくなりました。
# search_handler.py 側で timeout を渡している箇所は、
# generate_lowload_response を呼び出す側で timeout を直接設定することになります。
# 例：summarized_results_raw = await llm_manager.generate_lowload_response(summarize_prompt, timeout=300.0)