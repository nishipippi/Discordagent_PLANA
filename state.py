from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    user_input: str = Field(default="")
    chat_history: List[BaseMessage] = Field(default_factory=list)
    channel_id: int = Field(...)
    thread_id: Optional[int] = Field(default=None)
    search_query: Optional[str] = Field(default=None)
    search_results: Optional[List[Dict]] = Field(default=None)
    should_search_decision: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
