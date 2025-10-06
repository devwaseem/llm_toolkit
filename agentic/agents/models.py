from pydantic import BaseModel

from llm_toolkit.llm.models import LLMResponse


class AgentResponse(BaseModel):
    session_id: str
    answer: LLMResponse
