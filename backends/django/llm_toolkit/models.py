from django.db import models

from llm_toolkit.agentic.session.base import (
    AgentSession,
    AgentSessionReply,
    AgentSessionTransaction,
)
from llm_toolkit.llm.models import LLMInputMessage


class DjangoAgentSession(models.Model):
    id = models.AutoField(primary_key=True)
    objects: models.Manager["DjangoAgentSession"]

    agent_id = models.CharField(max_length=100)
    reply_to = models.JSONField(null=True, default=None, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    transaction_stack = models.JSONField(default=list)
    conversation_history = models.JSONField(default=list)
    running_tools = models.JSONField(default=list)

    class Meta:
        db_table = "agent_session"

    def __str__(self) -> str:
        return str(f"{self.id}: {self.agent_id}")

    def to_agent_session(self) -> AgentSession:
        reply_to = None
        if self.reply_to:
            reply_to = AgentSessionReply.model_validate(self.reply_to)

        session = AgentSession(
            session_id=str(self.id),
            agent_id=self.agent_id,
            reply_to=reply_to,
        )
        session.transaction_stack = [
            AgentSessionTransaction.model_validate(m)
            for m in self.transaction_stack
        ]
        session.conversation_history = [
            LLMInputMessage.model_validate(m)
            for m in self.conversation_history
        ]
        session.running_tools = set(self.running_tools)
        return session

    def apply(self, session: AgentSession) -> None:
        self.agent_id = session.agent_id
        self.reply_to = (
            session.reply_to.model_dump(mode="json")
            if session.reply_to
            else None
        )
        self.transaction_stack = [
            m.model_dump(mode="json") for m in session.transaction_stack
        ]
        self.conversation_history = [
            m.model_dump(mode="json") for m in session.conversation_history
        ]
        self.running_tools = list(session.running_tools)
