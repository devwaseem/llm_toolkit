import importlib
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypedDict

from gevent.pool import Pool as GeventPool
from jinja2 import Environment
from rich.console import Console
from rich.logging import RichHandler

from llm_toolkit.agentic.agents.models import AgentResponse
from llm_toolkit.agentic.runtime.broker import (
    AgentRuntimeTaskBroker,
    InMemoryAgentRuntimeTaskBroker,
)
from llm_toolkit.agentic.runtime.event import AgentRuntimeEvent
from llm_toolkit.agentic.session.base import AgentSession
from llm_toolkit.agentic.session.repo import AgentSessionRepository
from llm_toolkit.llm.models import (
    LLMStopReason,
    LLMToolCallRequest,
    LLMToolCallResponse,
)
from llm_toolkit.tool import LLMTool

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
        )
    ],
)
logger = logging.getLogger("Agent Runtime")

if TYPE_CHECKING:
    from llm_toolkit.agentic.agents.base import Agent

console = Console()


AgentName = str
ToolName = str


class RuntimeRegistryItem(TypedDict):
    agent: "Agent"
    tools: dict[ToolName, LLMTool]


_jinja2 = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
)

_INTRO_TEMPLATE = """
[bold green]Agent Runtime
-------------[/]
[bold magenta][Config][/]
> Queues: {{queues}}
> Concurrency: {{ concurrency}}
> Broker: {{ broker }}
> Session Repository: {{ session_repository }}

[bold magenta][Agents][/]
{% for r in registry %}
[magenta]• {{ r.agent.name }}[/]
    {% for tool in r['tools'].values() %}
    [cyan]• {{ tool.name }}[/]: [magenta]{{ tool.definition.short_description }}[/]
    {% endfor %}
{% else %}
[bold red]No Agent was registered[/]
{% endfor %}
"""


class AgentRuntimeRegistry:
    def __init__(self) -> None:
        self.registry: dict[str, RuntimeRegistryItem] = {}

    def get_agent(self, agent_id: str) -> "Agent":
        agent = self.registry[agent_id]["agent"]
        if not agent:
            raise ValueError(f"Agent {agent_id} is not registered")
        return agent

    def get_tool_from_agent(self, agent_id: str, tool_name: str) -> LLMTool:
        agent_tools = self._get_agent_tools(agent_id=agent_id)
        tool = agent_tools.get(tool_name)
        if not tool:
            raise ValueError(
                f"Tool: {tool_name} is not registered for agent: {agent_id}"
            )
        return tool

    def _get_agent_tools(self, agent_id: str) -> dict[str, LLMTool]:
        return self.registry[agent_id]["tools"]

    def register_agent(self, *, agent: "Agent") -> None:
        self.registry[agent.agent_id] = {
            "agent": agent,
            "tools": {},
        }
        for tool in agent.get_llm_tools():
            self._register_agent_tool(agent=agent, tool=tool)

    def _register_agent_tool(
        self,
        agent: "Agent",
        tool: Callable[..., str] | LLMTool,
    ) -> None:
        llm_tool: LLMTool
        if isinstance(tool, LLMTool):
            llm_tool = tool
        elif callable(tool):
            llm_tool = LLMTool.from_callable(tool)
        else:
            raise TypeError(f"Unsupported tool type: {tool}")

        if not llm_tool.func:
            raise ValueError(
                f"Tool {llm_tool.name} from agent {agent.name} is not callable"
            )

        self.registry[agent.agent_id]["tools"][llm_tool.name] = llm_tool

    def __getstate__(self) -> object:
        return self.registry

    def __setstate__(self, state: object) -> None:
        self.registry = state  # type: ignore


class RuntimeScheduler:
    def __init__(self, *, broker: AgentRuntimeTaskBroker) -> None:
        self.broker = broker

    def schedule_agent_event(
        self,
        *,
        session_id: str,
        addtional_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.broker.notify(
            event=AgentRuntimeEvent(
                task_type="agent",
                session_id=session_id,
                additional_context=addtional_context,
                metadata=metadata,
            ),
            queue="default",
        )

    def schedule_tool_call(
        self,
        *,
        request: LLMToolCallRequest,
        session_id: str,
        additional_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "Session[%s]: Tool[%s]: Scheduling...",
            session_id,
            request.name,
        )

        self.broker.notify(
            event=AgentRuntimeEvent(
                task_type="tool",
                session_id=session_id,
                tool_call_request=request,
                additional_context=additional_context,
                metadata=metadata,
            ),
            queue="default",
        )


def handle_event_gracefully(
    session_repository: AgentSessionRepository,
    scheduler: RuntimeScheduler,
    registry: AgentRuntimeRegistry,
    event: AgentRuntimeEvent,
) -> None:
    try:
        handle_event(
            session_repository=session_repository,
            scheduler=scheduler,
            registry=registry,
            event=event,
        )
    except Exception:
        logger.exception(
            "Session[%s]: Failed to handle event: %s",
            event.session_id,
            event.model_dump_json(),
        )
        with session_repository.select_for_update(
            session_id=event.session_id
        ) as s:
            s.rollback_transaction()
            s.add_ai_answer(
                answer=(
                    "I'm sorry, an issue occurred "
                    "while processing your request. Our team "
                    "has been notified and is looking into it. Please try "
                    "again in a moment."
                ),
            )


def handle_event(
    *,
    session_repository: AgentSessionRepository,
    scheduler: RuntimeScheduler,
    registry: AgentRuntimeRegistry,
    event: AgentRuntimeEvent,
) -> None:
    session = session_repository.get(session_id=event.session_id)

    logger.info(
        "Session[%s]: Running event %s",
        session.id,
        event.model_dump_json(),
    )

    if event.task_type == "agent":
        handle_agent_event(
            session_repository=session_repository,
            scheduler=scheduler,
            registry=registry,
            session=session,
            additional_context=event.additional_context,
            metadata=event.metadata,
        )

    elif event.task_type == "tool" and (
        tool_request := event.tool_call_request
    ):
        handle_tool_event(
            session_repository=session_repository,
            scheduler=scheduler,
            registry=registry,
            session=session,
            tool_request=tool_request,
            additional_context=event.additional_context,
            metadata=event.metadata,
        )


def handle_tool_event(
    *,
    session_repository: AgentSessionRepository,
    scheduler: RuntimeScheduler,
    registry: AgentRuntimeRegistry,
    session: AgentSession,
    tool_request: LLMToolCallRequest,
    additional_context: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    tool = registry.get_tool_from_agent(
        agent_id=session.agent_id, tool_name=tool_request.name
    )

    assert tool.func is not None

    tool_arguments = tool_request.arguments
    if tool.is_metadata_requested:
        tool_arguments["metadata"] = metadata

    try:
        result = tool.func(**tool_arguments)
    except Exception:
        # Catch any unhandled exceptions raised by the tool function
        # and return a generic error message
        logger.exception(
            "Session[%s]: Tool[%s]: Error executing tool",
            session.id,
            tool.name,
            stack_info=False,
        )
        result = "An error occurred while executing the tool"

    llm_tool_call_response = LLMToolCallResponse(
        tool_call=tool_request,
        output=result,
    )
    logger.debug(
        "Session[%s]: Tool[%s]: %s",
        session.id,
        tool.name,
        llm_tool_call_response,
    )

    with session_repository.select_for_update(session_id=session.id) as s:
        s.add_tool_call_response(
            response=llm_tool_call_response,
        )

    scheduler.schedule_agent_event(
        session_id=session.id,
        metadata=metadata,
        addtional_context=additional_context,
    )


def handle_agent_event(
    *,
    session_repository: AgentSessionRepository,
    scheduler: RuntimeScheduler,
    registry: AgentRuntimeRegistry,
    session: AgentSession,
    additional_context: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    with session_repository.select_for_update(session_id=session.id) as s:
        pending_tool_calls = s.get_pending_tool_calls()
        if pending_tool_calls > 0:
            logger.info(
                "Session[%s]: Has %s pending tool calls, waiting...",
                s.id,
                pending_tool_calls,
            )
            return

    agent = registry.get_agent(agent_id=session.agent_id)
    if not agent:
        raise ValueError(f"{session.agent_id} is not registered")

    logger.info("Session[%s]: Agent[%s]: Running...", session.id, agent.name)
    llm_response = agent.run(
        session=session,
        additional_context=additional_context,
        metadata=metadata,
    )
    logger.info(
        "Session[%s]: Agent[%s]: %s",
        session.id,
        agent.name,
        llm_response.stop_reason,
    )
    logger.debug(
        "Session[%s]: Agent[%s]: %s",
        session.id,
        agent.name,
        llm_response,
    )

    if llm_response.stop_reason == LLMStopReason.TOOL_USE:
        if not llm_response.function_calls:
            # might have handled in higher order components
            return

        for tool_call_request in llm_response.function_calls:
            with session_repository.select_for_update(
                session_id=session.id
            ) as s:
                s.add_tool_call_request(
                    request=tool_call_request,
                )
            scheduler.schedule_tool_call(
                request=tool_call_request,
                session_id=session.id,
                metadata=metadata,
                additional_context=additional_context,
            )

    elif llm_response.stop_reason == LLMStopReason.END_TURN:
        with session_repository.select_for_update(session_id=session.id) as s:
            s.add_ai_answer(
                answer=llm_response.answer,
            )

        # Only call on_response if it is meant for the user
        if agent.on_response and session.reply_to is None:
            agent.on_response(
                AgentResponse(session_id=session.id, answer=llm_response)
            )

        if session.reply_to:
            with session_repository.select_for_update(
                session_id=session.reply_to.session_id
            ) as parent_session:
                parent_session.add_tool_call_response(
                    response=LLMToolCallResponse(
                        output=llm_response.answer,
                        tool_call=session.reply_to.tool_request,
                    ),
                )

            scheduler.schedule_agent_event(
                session_id=parent_session.id,
                metadata=metadata,
                addtional_context=additional_context,
            )

    else:
        raise NotImplementedError(f"Stop reason: {llm_response.stop_reason}")


class AgentRuntime:
    registry: AgentRuntimeRegistry

    def __init__(
        self,
        *,
        broker: AgentRuntimeTaskBroker | None = None,
        session_repository: AgentSessionRepository,
    ) -> None:
        self.registry = AgentRuntimeRegistry()

        self.broker = broker or InMemoryAgentRuntimeTaskBroker()
        self.scheduler = RuntimeScheduler(broker=self.broker)

        self.session_repository = session_repository
        self.should_process_events = True

    def import_modules(self, modules: list[str]) -> None:
        for module in modules:
            importlib.import_module(module)

    def register_agent(self, *, agent: "Agent") -> None:
        self.registry.register_agent(agent=agent)

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

    def start(
        self,
        *,
        queues: Sequence[str],
        pool: Literal["prefork", "gevent", "threads"],
        workers: int,
    ) -> None:
        console.print(
            _jinja2.from_string(_INTRO_TEMPLATE).render(
                concurrency=f"{workers} ({pool})",
                queues=", ".join(queues),
                broker=self.broker.get_display_name(),
                session_repository=self.session_repository.get_display_name(),
                registry=self.registry.registry.values(),
            )
        )
        runtime_pool: ProcessPoolExecutor | GeventPool | ThreadPoolExecutor
        match pool:
            case "prefork":
                runtime_pool = ProcessPoolExecutor(max_workers=workers)
            case "gevent":
                runtime_pool = GeventPool(size=workers)
            case "threads":
                runtime_pool = ThreadPoolExecutor(max_workers=workers)

        logger.info("Listening for events...")
        try:
            while self.should_process_events:
                event = self.broker.get_event(queues=queues or ["default"])
                logger.info("Received event: %s", event.model_dump_json())
                if isinstance(
                    runtime_pool, (ThreadPoolExecutor, ProcessPoolExecutor)
                ):
                    runtime_pool.submit(
                        handle_event_gracefully,
                        session_repository=self.session_repository,
                        scheduler=self.scheduler,
                        registry=self.registry,
                        event=event,
                    )
                elif isinstance(runtime_pool, GeventPool):
                    runtime_pool.spawn(
                        handle_event_gracefully,
                        session_repository=self.session_repository,
                        scheduler=self.scheduler,
                        registry=self.registry,
                        event=event,
                    )
        except KeyboardInterrupt:
            console.print("\n[bold red]Shutting down...[/]")
            self._shutdown(pool=runtime_pool)

    def _shutdown(
        self, *, pool: ProcessPoolExecutor | GeventPool | ThreadPoolExecutor
    ) -> None:
        self.should_process_events = False
        if isinstance(pool, (ThreadPoolExecutor, ProcessPoolExecutor)):
            try:
                pool.shutdown(wait=True)
            except KeyboardInterrupt:
                pool.shutdown(wait=False)
        elif isinstance(pool, GeventPool):
            try:
                pool.join()
            except KeyboardInterrupt:
                pool.kill()
