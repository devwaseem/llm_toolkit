from contextlib import contextmanager
from typing import Any, Generator, override

import django
from django.apps import apps
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady
from django.db import transaction

from llm_toolkit.agentic.session.base import AgentSession, AgentSessionReply
from llm_toolkit.agentic.session.repo import AgentSessionRepository


class DjangoAgentSessionRepository(AgentSessionRepository):
    def __init__(self) -> None:
        # try to setup the django if it's not already setup
        try:
            apps.check_models_ready()
        except AppRegistryNotReady:
            django.setup()

        # Lazy import to avoid importing it before django is ready
        from llm_toolkit.backends.django.llm_toolkit.models import (
            DjangoAgentSession,
        )

        self.model = DjangoAgentSession

    @override
    def create_session(
        self,
        *,
        agent_id: str,
        reply_to: AgentSessionReply | None = None,
    ) -> AgentSession:
        session = self.model.objects.create(
            agent_id=agent_id,
            reply_to=reply_to.model_dump(mode="json") if reply_to else None,
        )
        return session.to_agent_session()

    @override
    def get_display_name(self) -> str:
        return f"django ({settings.DATABASES['default']['ENGINE']})"

    @override
    def get(self, *, session_id: str) -> AgentSession:
        session = self.model.objects.get(id=int(session_id))
        return session.to_agent_session()

    @override
    def save(self, *, session: AgentSession) -> None:
        obj = self.model.objects.get(id=session.id)
        obj.apply(session=session)
        obj.save()

    @override
    @contextmanager
    def select_for_update(
        self, *, session_id: str
    ) -> Generator[AgentSession, None, None]:
        with transaction.atomic():
            obj = self.model.objects.select_for_update().get(id=session_id)
            session = obj.to_agent_session()
            yield session
            self.save(session=session)

    @override
    def get_init_kwargs(self) -> dict[str, Any]:
        return {}
