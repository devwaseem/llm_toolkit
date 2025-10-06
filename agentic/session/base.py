from typing import Generator
from uuid import uuid4

from pydantic import BaseModel

from llm_toolkit.agentic.session.exceptions import (
    TransactionAlreadyFinishedError,
    TransactionAlreadyRunningError,
    TransactionNotStartedError,
)
from llm_toolkit.llm.models import (
    LLMInputMessage,
    LLMToolCallRequest,
    LLMToolCallResponse,
)


class AgentSessionReply(BaseModel):
    session_id: str
    tool_request: LLMToolCallRequest


class AgentSessionTransaction(BaseModel):
    id: str
    query: str
    tool_stack: list[LLMToolCallRequest | LLMToolCallResponse]
    answer: str | None = None

    def pending_tool_calls(self) -> int:
        pending_count = 0
        for tool_call in self.tool_stack:
            if isinstance(tool_call, LLMToolCallRequest):
                pending_count += 1
            elif isinstance(tool_call, LLMToolCallResponse):
                pending_count -= 1
        return pending_count


class AgentSession:
    transaction_stack: list[AgentSessionTransaction]
    conversation_history: list[LLMInputMessage]
    agent_context: str

    class NotFoundError(Exception):
        pass

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> None:
        self.id = session_id
        self.agent_id = agent_id
        self.reply_to = reply_to
        self.transaction_stack = []
        self.conversation_history = []
        self.agent_context = ""

    @property
    def _last_transaction(self) -> AgentSessionTransaction:
        if not self.transaction_stack:
            raise TransactionNotStartedError

        return self.transaction_stack[-1]

    def get_pending_tool_calls(self) -> int:
        return self._last_transaction.pending_tool_calls()

    def validate_run(self) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

    def add_tool_call_request(
        self,
        *,
        request: LLMToolCallRequest,
    ) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

        self._last_transaction.tool_stack.append(request)

    def add_tool_call_response(self, *, response: LLMToolCallResponse) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

        self._last_transaction.tool_stack.append(response)

    def add_user_query(self, *, query: str) -> None:
        if (
            len(self.transaction_stack) > 0
            and not self._last_transaction.answer
        ):
            raise TransactionAlreadyRunningError

        self.transaction_stack.append(
            AgentSessionTransaction(
                id=uuid4().hex,
                query=query,
                tool_stack=[],
                answer=None,
            )
        )
        self.conversation_history.append(
            LLMInputMessage.from_human(content=query)
        )

    def add_ai_answer(self, *, answer: str) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

        self._last_transaction.answer = answer
        self.conversation_history.append(LLMInputMessage.from_ai(answer))

    @property
    def is_running(self) -> bool:
        return self._last_transaction.answer is None

    def rollback_transaction(self) -> None:
        if not self.transaction_stack:
            raise TransactionNotStartedError

        if self._last_transaction.answer is not None:
            raise TransactionAlreadyFinishedError

        self._last_transaction.tool_stack = []
        self._last_transaction.answer = None

    def to_llm_messages(self) -> Generator[LLMInputMessage, None, None]:
        for transaction in self.transaction_stack:
            yield LLMInputMessage.from_human(content=transaction.query)
            for tool_call in transaction.tool_stack:
                if isinstance(tool_call, LLMToolCallRequest):
                    yield LLMInputMessage.from_tool_call_request(
                        tool_call=tool_call,
                    )
                elif isinstance(tool_call, LLMToolCallResponse):
                    yield LLMInputMessage.from_tool_response(
                        response=tool_call,
                    )

            answer = transaction.answer
            if answer:
                yield LLMInputMessage.from_ai(answer)
