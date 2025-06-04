from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator, root_validator
import json # 追加

class AgentState(BaseModel):
    input_text: str = Field(default="") # user_input を input_text に変更
    chat_history: List[BaseMessage] = Field(default_factory=list)
    server_id: str = Field(...) # 新規追加
    channel_id: int = Field(...) # 型を int のままにする
    user_id: str = Field(...) # 新規追加
    thread_id: Optional[int] = Field(default=None) # 既存
    # --- For Tool Calling ---
    tool_name: Optional[str] = Field(default=None) # LLMが呼び出すと判断したツール名
    tool_args: Optional[Dict[str, Any]] = Field(default=None) # LLMが生成したツール引数
    tool_output: Optional[str] = Field(default=None) # ツールの実行結果
    # --- For Search (既存のものを統合または置換) ---
    search_query: Optional[str] = Field(default=None) # 既存
    search_results: Optional[List[Dict]] = Field(default=None) # 既存
    should_search_decision: Optional[str] = Field(default=None) # 既存
    # --- LLMの直接応答 ---
    llm_direct_response: Optional[str] = Field(default=None) # ツールを使わない場合のLLMの応答

    @field_validator('chat_history', mode='before')
    @classmethod
    def validate_chat_history(cls, v: List[Any]) -> List[BaseMessage]:
        converted_messages = []
        for item in v:
            if isinstance(item, dict):
                # 辞書形式の場合、type フィールドを見て適切なメッセージクラスに変換
                msg_type = item.get('type')
                content = item.get('content') # content は Any 型または None になりうる

                if content is None:
                    # content が None の場合はスキップするか、エラーを発生させる
                    print(f"Warning: Skipping message with missing content during chat_history validation: {item}")
                    continue
                
                # content が文字列であることを保証
                if not isinstance(content, str):
                    content = str(content) # 強制的に文字列に変換

                if msg_type == 'human':
                    converted_messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    converted_messages.append(AIMessage(content=content))
                elif msg_type == 'system':
                    converted_messages.append(SystemMessage(content=content))
                elif msg_type == 'tool':
                    tool_call_id = item.get('tool_call_id', 'unknown_tool_call_id')
                    converted_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                else:
                    # 未知のタイプの場合、content があれば BaseMessage として追加
                    converted_messages.append(BaseMessage(content=content))
            elif isinstance(item, BaseMessage):
                converted_messages.append(item)
            else:
                print(f"Warning: Skipping unexpected non-BaseMessage type in chat_history during validation: {type(item)}, Value: {item}")
        return converted_messages

    class Config:
        arbitrary_types_allowed = True

class ToolCall(BaseModel):
    name: str = Field(description="呼び出すツールの名前")
    args: Union[Dict[str, Any], str] = Field(description="ツールに渡す引数の辞書またはJSON文字列") # 型を Union に変更

    @field_validator('args', mode='before') # mode='before' でPydanticの型変換前に実行
    @classmethod
    def parse_args_if_str(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                # パースに失敗した場合、エラーを発生させるか、
                # あるいは空の辞書を返すなどのフォールバック処理も検討できる
                raise ValueError(f"Failed to parse args string to dict: {value}, Error: {e}")
        elif isinstance(value, dict):
            return value # 既に辞書ならそのまま返す
        # 予期しない型の場合はエラー
        raise TypeError(f"args must be a dict or a valid JSON string, got {type(value)}")

class LLMDecisionOutput(BaseModel):
    thought: str = Field(description="LLMの思考プロセス。なぜその判断に至ったかを記述する。")
    tool_call: Optional[ToolCall] = Field(None, description="ツールを呼び出す場合に設定。direct_response とは排他的。")
    direct_response: Optional[str] = Field(None, description="ツールを呼び出さず直接応答する場合に設定。tool_call とは排他的。空であってはならない。")

    @root_validator(pre=False, skip_on_failure=True) # pre=False でフィールドバリデーション後に実行
    @classmethod
    def check_tool_call_or_direct_response_exists(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        tool_call = values.get('tool_call')
        direct_response = values.get('direct_response')

        if tool_call is None and direct_response is None:
            raise ValueError("LLMDecisionOutput must have either 'tool_call' or 'direct_response'. Both are None.")
        if tool_call is not None and direct_response is not None:
            raise ValueError("'tool_call' and 'direct_response' are mutually exclusive. Both have values.")
        
        # direct_response が None でなく、かつ空文字列でないことを確認 (スキーマの description に合わせて)
        if direct_response is not None and not direct_response.strip():
             raise ValueError("'direct_response' cannot be an empty string or only whitespace.")

        return values
