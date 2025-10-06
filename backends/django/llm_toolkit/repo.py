from contextlib import contextmanager
from typing import Generator, override

from django.db import transaction

from llm_toolkit.agentic.session.base import AgentSession, AgentSessionReply
from llm_toolkit.agentic.session.repo import AgentSessionRepository
from llm_toolkit.backends.django.llm_toolkit.models import DjangoAgentSession


class DjangoAgentSessionRepository(AgentSessionRepository):
    @override
    def create_session(
        self,
        *,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> AgentSession:
        session = DjangoAgentSession.objects.create(
            agent_id=agent_id,
            reply_to=reply_to.model_dump(mode="json") if reply_to else None,
        )
        return session.to_agent_session()

    @override
    def get_display_name(self) -> str:
        return "Django"

    @override
    def get(self, *, session_id: str) -> AgentSession:
        session = DjangoAgentSession.objects.get(id=int(session_id))
        return session.to_agent_session()

    @override
    def save(self, *, session: AgentSession) -> None:
        obj = DjangoAgentSession.objects.get(id=session.id)
        obj.apply(session=session)
        obj.save()

    @override
    @contextmanager
    def select_for_update(
        self, *, session_id: str
    ) -> Generator[AgentSession, None, None]:
        with transaction.atomic():
            obj = DjangoAgentSession.objects.select_for_update().get(
                id=session_id
            )
            session = obj.to_agent_session()
            yield session
            self.save(session=session)
