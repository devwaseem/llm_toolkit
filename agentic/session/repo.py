from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, override
from uuid import uuid4

from llm_toolkit.agentic.session.base import AgentSession, AgentSessionReply


class AgentSessionRepository(ABC):
    @abstractmethod
    def get_display_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, *, session_id: str) -> AgentSession:
        raise NotImplementedError

    @abstractmethod
    def save(self, *, session: AgentSession) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_session(
        self,
        *,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> AgentSession:
        raise NotImplementedError

    @contextmanager
    def select_for_update(
        self, *, session_id: str
    ) -> Generator[AgentSession, None, None]:
        session = self.get(session_id=session_id)
        yield session
        self.save(session=session)


class InMemoryAgentSessionRepository(AgentSessionRepository):
    def __init__(self) -> None:
        self.sessions: dict[str, AgentSession] = {}

    @override
    def get_display_name(self) -> str:
        return "In-Memory"

    @override
    def get(self, *, session_id: str) -> AgentSession:
        if session_id not in self.sessions:
            raise AgentSession.NotFoundError(f"Session {session_id} not found")
        return self.sessions[session_id]

    @override
    def save(self, *, session: AgentSession) -> None:
        self.sessions[session.id] = session

    @override
    def create_session(
        self,
        *,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> AgentSession:
        session = AgentSession(
            session_id=uuid4().hex,
            agent_id=agent_id,
            reply_to=reply_to,
        )
        self.save(session=session)
        return session
