from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_toolkit.agentic.runtime.base import RuntimeScheduler
    from llm_toolkit.agentic.session.repo import AgentSessionRepository


class AgentRunner:
    def __init__(
        self,
        *,
        session_repository: "AgentSessionRepository",
        scheduler: "RuntimeScheduler",
    ) -> None:
        self.session_repository = session_repository
        self.scheduler = scheduler

    def query_agent(
        self,
        *,
        session_id: str,
        query: str,
        additional_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.session_repository.select_for_update(
            session_id=session_id
        ) as session:
            session.add_user_query(query=query)

        session.validate_run()
        self.scheduler.schedule_agent_event(
            session_id=session.id,
            addtional_context=additional_context,
            metadata=metadata,
        )

    def run_session(
        self,
        *,
        session_id: str,
        additional_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.scheduler.schedule_agent_event(
            session_id=session_id,
            addtional_context=additional_context,
            metadata=metadata,
        )
