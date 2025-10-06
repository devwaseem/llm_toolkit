from typing import Any, Literal

from pydantic import BaseModel

from llm_toolkit.llm.models import LLMToolCallRequest


class AgentRuntimeEvent(BaseModel):
    task_type: Literal["agent", "tool"]
    session_id: str
    metadata: dict[str, Any] | None = None
    additional_context: str | None = None
    tool_call_request: LLMToolCallRequest | None = None
