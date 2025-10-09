from collections import defaultdict
from datetime import datetime, timezone
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


class AgentSessionTransactionToolRequest(BaseModel):
    created_at: datetime
    tool_request: LLMToolCallRequest


class AgentSessionTransactionToolResponse(BaseModel):
    created_at: datetime
    tool_response: LLMToolCallResponse
    from_session_id: str | None = None


class AgentSessionTransaction(BaseModel):
    id: str
    queried_at: datetime
    query: str
    tool_stack: list[
        AgentSessionTransactionToolRequest
        | AgentSessionTransactionToolResponse
    ]
    answer: str | None = None
    answered_at: datetime | None = None

    @property
    def pending_tool_calls_count(self) -> int:
        return len(self.get_pending_tool_calls())

    @property
    def has_pending_tool_calls(self) -> bool:
        return self.pending_tool_calls_count > 0

    def get_pending_tool_calls(self) -> set[str]:
        pending_tools = set()
        for tool_call in self.tool_stack:
            if isinstance(tool_call, AgentSessionTransactionToolRequest):
                pending_tools.add(tool_call.tool_request.name)
            elif isinstance(tool_call, AgentSessionTransactionToolResponse):
                pending_tools.remove(tool_call.tool_response.tool_call.name)
        return pending_tools

    def get_sorted_tool_stack(
        self,
    ) -> Generator[
        AgentSessionTransactionToolRequest
        | AgentSessionTransactionToolResponse,
        None,
        None,
    ]:
        tool_map = defaultdict[
            str,
            list[
                AgentSessionTransactionToolRequest
                | AgentSessionTransactionToolResponse,
            ],
        ](list)
        for tool in self.tool_stack:
            if isinstance(tool, AgentSessionTransactionToolRequest):
                tool_map[tool.tool_request.name].append(tool)
            elif isinstance(tool, AgentSessionTransactionToolResponse):
                tool_map[tool.tool_response.tool_call.name].append(tool)

        for tools in tool_map.values():
            for tool in tools:
                yield tool


class AgentMessage(BaseModel):
    created_at: datetime
    message: LLMInputMessage


class AgentSession:
    transaction_stack: list[AgentSessionTransaction]
    conversation_history: list[AgentMessage]
    agent_context: str
    running_tools: set[str]

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
        self.running_tools = set()
        self.agent_context = ""

    @property
    def _last_transaction(self) -> AgentSessionTransaction:
        if not self.transaction_stack:
            raise TransactionNotStartedError

        return self.transaction_stack[-1]

    def get_pending_tool_calls_count(self) -> int:
        return self._last_transaction.pending_tool_calls_count

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

        self._last_transaction.tool_stack.append(
            AgentSessionTransactionToolRequest(
                created_at=datetime.now(tz=timezone.utc),
                tool_request=request,
            )
        )

    def add_tool_call_response(
        self,
        *,
        response: LLMToolCallResponse,
        from_session_id: str | None = None,
    ) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

        self._last_transaction.tool_stack.append(
            AgentSessionTransactionToolResponse(
                created_at=datetime.now(tz=timezone.utc),
                tool_response=response,
                from_session_id=from_session_id,
            )
        )

    def add_user_query(self, *, query: str) -> None:
        if (
            len(self.transaction_stack) > 0
            and not self._last_transaction.answer
        ):
            raise TransactionAlreadyRunningError

        date_now = datetime.now(tz=timezone.utc)
        self.transaction_stack.append(
            AgentSessionTransaction(
                id=uuid4().hex,
                queried_at=date_now,
                query=query,
                tool_stack=[],
                answer=None,
                answered_at=None,
            )
        )
        self.conversation_history.append(
            AgentMessage(
                created_at=date_now,
                message=LLMInputMessage.from_human(content=query),
            )
        )

    def add_ai_answer(self, *, answer: str) -> None:
        if self._last_transaction.answer:
            raise TransactionAlreadyFinishedError

        date_now = datetime.now(tz=timezone.utc)
        self._last_transaction.answer = answer
        self._last_transaction.answered_at = date_now
        self.conversation_history.append(
            AgentMessage(
                created_at=date_now,
                message=LLMInputMessage.from_ai(answer),
            )
        )

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

    def to_agent_messages(self) -> Generator[AgentMessage, None, None]:
        for transaction in self.transaction_stack:
            yield AgentMessage(
                created_at=transaction.queried_at,
                message=LLMInputMessage.from_human(content=transaction.query),
            )
            for tool in transaction.get_sorted_tool_stack():
                if isinstance(tool, AgentSessionTransactionToolRequest):
                    yield AgentMessage(
                        created_at=tool.created_at,
                        message=LLMInputMessage.from_tool_call_request(
                            tool_call=tool.tool_request,
                        ),
                    )
                if isinstance(tool, AgentSessionTransactionToolResponse):
                    yield AgentMessage(
                        created_at=tool.created_at,
                        message=LLMInputMessage.from_tool_response(
                            response=tool.tool_response,
                        ),
                    )

            if answer := transaction.answer:
                assert transaction.answered_at is not None  # nosec
                yield AgentMessage(
                    created_at=transaction.answered_at,
                    message=LLMInputMessage.from_ai(answer),
                )
