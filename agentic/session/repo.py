from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, override
from uuid import uuid4

import orjson
import redis

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

    @abstractmethod
    def get_init_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryAgentSessionRepository(AgentSessionRepository):
    def __init__(self) -> None:
        self.sessions: dict[str, AgentSession] = {}
        self.next_session_id = 0

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
            session_id=str(self.next_session_id),
            agent_id=agent_id,
            reply_to=reply_to,
        )
        self.save(session=session)
        self.next_session_id += 1
        return session

    @override
    def get_init_kwargs(self) -> dict[str, Any]:
        return {}


class RedisAgentSessionRepository(AgentSessionRepository):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )

    @override
    def get_display_name(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"

    def get_computed_id(self, session_id: str) -> str:
        return f"agentic:session:{session_id}"

    @override
    def get(self, *, session_id: str) -> AgentSession:
        session_json = self.redis.get(
            self.get_computed_id(session_id),
        )
        if not session_json:
            raise AgentSession.NotFoundError(f"Session {session_id} not found")
        return AgentSession.from_dict(orjson.loads(session_json))  # type: ignore

    @override
    def save(self, *, session: AgentSession) -> None:
        self.redis.set(
            self.get_computed_id(session.id),
            orjson.dumps(session.to_dict()),
        )

    @override
    def create_session(
        self,
        *,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> AgentSession:
        session = AgentSession(
            session_id=str(uuid4()),
            agent_id=agent_id,
            reply_to=reply_to,
        )
        self.save(session=session)
        return session

    @override
    def get_init_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
        }
