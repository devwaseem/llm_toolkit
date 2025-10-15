import importlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
    TypedDict,
    cast,
)

from gevent.pool import Pool as GeventPool
from jinja2 import Environment
from rich.console import Console

from llm_toolkit.agentic.runner import AgentRunner
from llm_toolkit.agentic.runtime.broker import (
    AgentRuntimeTaskBroker,
    InMemoryAgentRuntimeTaskBroker,
)
from llm_toolkit.agentic.runtime.event import AgentRuntimeEvent
from llm_toolkit.agentic.session.base import AgentSession
from llm_toolkit.agentic.session.repo import (
    AgentSessionRepository,
    InMemoryAgentSessionRepository,
)
from llm_toolkit.llm.models import (
    LLMStopReason,
    LLMToolCallRequest,
    LLMToolCallResponse,
)
from llm_toolkit.tool import LLMTool

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
===================================================================
>> MULTI AGENT RUNTIME <<
===================================================================

█ CONFIG
───────────────────────────────────────────────────────────────────
├─ Queues             : {{queues}}
├─ Concurrency        : {{ concurrency}}
├─ Broker             : {{ broker }}
└─ Session Repository : {{ session_repository }}

█ AGENT INVENTORY ({{ agents_count }})
───────────────────────────────────────────────────────────────────
{% for i in agents %}
[magenta]► {% if i.is_supervisor %} 👑 {% endif %}{{ i.name }} [/]
  └ Role: {{ i.role }}
{% if i.team %}
  └ Team:
    {% for team_member in i.team %}
    [cyan]└ {{ team_member }}[/]
    {% endfor %}
{% endif %}
{% if i.tools %}
  └ Tools:
    {% for tool_name, tool_definition in i.tools.items() %}
    [cyan]└ {{ tool_name }}[/]: [green]{{ tool_definition }}[/]
    {% endfor %}
{% endif %}

{% else %}
[bold red]No Agent was registered[/]
{% endfor %}
"""


def _render_header(
    workers: int,
    pool: str,
    queues: Sequence[str],
    broker: str,
    session_repository: str,
    registry: list[RuntimeRegistryItem],
) -> None:
    from llm_toolkit.agentic.agents.supervisor import SupervisorAgent

    agents: list[dict[str, Any]] = []

    for r in registry:
        data: dict[str, Any] = {
            "team": [],
            "is_supervisor": False,
        }
        agent = r["agent"]
        is_supervisor = isinstance(r["agent"], SupervisorAgent)
        data["name"] = agent.name
        data["role"] = agent.role
        data["is_supervisor"] = is_supervisor
        data["tools_count"] = len(r["agent"].get_llm_tools())
        data["tools"] = {
            tool.name: tool.definition.short_description
            for tool in r["agent"].get_llm_tools()
        }
        if is_supervisor:
            supervisor_agent = cast(SupervisorAgent, agent)
            team_agents = list(supervisor_agent.team.values())
            for t in team_agents:
                data["team"].append(t.name)
        agents.append(data)

    agents.sort(key=lambda a: a["is_supervisor"], reverse=True)

    template = _jinja2.from_string(_INTRO_TEMPLATE).render(
        concurrency=f"{workers} ({pool})",
        queues=", ".join(queues),
        broker=broker,
        session_repository=session_repository,
        agents=agents,
        agents_count=len(agents),
    )
    console.print(template)


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

    def register_agent_from_path(self, path: str) -> None:
        from llm_toolkit.agentic.agents.base import Agent

        paths = path.split(".")
        module_path = ".".join(paths[:-1])
        obj_name = paths[-1]
        module = importlib.import_module(module_path)
        agent_obj = getattr(module, obj_name)
        if not isinstance(agent_obj, Agent):
            raise TypeError(
                f"Imported object {obj_name} is not an instance of Agent"
            )
        self.register_agent(agent=agent_obj)

    def register_agent(self, *, agent: "Agent") -> None:
        from llm_toolkit.agentic.agents.supervisor import SupervisorAgent

        self.registry[agent.agent_id] = {
            "agent": agent,
            "tools": {},
        }
        for tool in agent.get_llm_tools():
            self._register_agent_tool(agent=agent, tool=tool)

        if isinstance(agent, SupervisorAgent):
            for team_member in agent.team.values():
                self.register_agent(agent=team_member)

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


_GLOBAL_STATE: dict[str, Any] = {}


# Only used for multiprocessing.ProcessPoolExecutor because it has
# different memory space for each worker process
def _initializer_worker(
    scheduler_cls: type[RuntimeScheduler],
    broker_cls: type[AgentRuntimeTaskBroker],
    broker_kwargs: dict[str, Any],
    session_repo_cls: type[AgentSessionRepository],
    session_repo_kwargs: dict[str, Any],
    agent_paths: list[str],
    on_setup: Callable[..., None] | None = None,
) -> None:
    if on_setup is not None:
        on_setup()

    broker = broker_cls(**broker_kwargs)
    scheduler = scheduler_cls(broker=broker)

    session_repository = session_repo_cls(**session_repo_kwargs)

    registry = AgentRuntimeRegistry()
    for agent_path in agent_paths:
        registry.register_agent_from_path(path=agent_path)

    global _GLOBAL_STATE
    _GLOBAL_STATE = {
        "session_repository": session_repository,
        "scheduler": scheduler,
        "registry": registry,
    }


def _handle_process_pool_worker_event(event: AgentRuntimeEvent) -> None:
    """
    The target function run by the pool. It retrieves dependencies from
    the process-local global state.
    """
    global _GLOBAL_STATE
    if not _GLOBAL_STATE:
        raise RuntimeError("Worker process was not initialized correctly.")

    # Get the dependencies initialized by the worker's initializer
    session_repository = _GLOBAL_STATE["session_repository"]
    scheduler = _GLOBAL_STATE["scheduler"]
    registry = _GLOBAL_STATE["registry"]

    handle_event_gracefully(
        session_repository=session_repository,
        scheduler=scheduler,
        registry=registry,
        event=event,
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

    tool_arguments_json = json.dumps(tool_arguments, indent=2)

    if tool.display_name:
        with session_repository.select_for_update(session_id=session.id) as s:
            s.running_tools.add(tool.display_name)

    logger.info(
        "Session[%s]: Tool[%s{%s}]: Running with params: %s",
        session.id,
        tool.name,
        tool.display_name,
        tool_arguments_json,
    )

    try:
        result = tool.func(**tool_arguments)
    except Exception:
        # Catch any unhandled exceptions raised by the tool function
        # and return a generic error message
        logger.exception(
            "Session[%s]: Tool[%s{%s}]: Error executing tool",
            session.id,
            tool.name,
            tool.display_name,
            stack_info=False,
        )
        result = "An error occurred while executing the tool"

    llm_tool_call_response = LLMToolCallResponse(
        tool_call=tool_request,
        output=result,
    )

    logger.debug(
        "Session[%s]: Tool[%s{%s}]: %s",
        session.id,
        tool.name,
        tool.display_name,
        llm_tool_call_response.model_dump_json(indent=2),
    )

    logger.info(
        "Session[%s]: Tool[%s{%s}]: Run Complete with arguments %s",
        session.id,
        tool.name,
        tool.display_name,
        tool_arguments_json,
    )

    with session_repository.select_for_update(session_id=session.id) as s:
        s.add_tool_call_response(
            response=llm_tool_call_response,
        )
        if tool.display_name in s.running_tools:
            s.running_tools.remove(tool.display_name)

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
        pending_tool_calls = s.get_pending_tool_calls_count()
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
        runner=AgentRunner(
            session_repository=session_repository,
            scheduler=scheduler,
        ),
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

        if session.reply_to:
            with session_repository.select_for_update(
                session_id=session.reply_to.session_id
            ) as parent_session:
                parent_session.add_tool_call_response(
                    response=LLMToolCallResponse(
                        output=llm_response.answer,
                        tool_call=session.reply_to.tool_request,
                    ),
                    from_session_id=session.id,
                )

            scheduler.schedule_agent_event(
                session_id=parent_session.id,
                metadata=metadata,
                addtional_context=additional_context,
            )

    else:
        raise NotImplementedError(f"Stop reason: {llm_response.stop_reason}")


class AgentRuntime:
    def __init__(
        self,
        *,
        broker: AgentRuntimeTaskBroker | None = None,
        session_repository: AgentSessionRepository | None = None,
        agent_paths: list[str],
        on_setup: Callable[..., None] | None = None,
    ) -> None:
        self.broker = broker if broker else InMemoryAgentRuntimeTaskBroker()
        self.scheduler = RuntimeScheduler(broker=self.broker)

        self.session_repository = (
            session_repository
            if session_repository
            else InMemoryAgentSessionRepository()
        )
        self.should_process_events = True
        self.runner = AgentRunner(
            session_repository=self.session_repository,
            scheduler=self.scheduler,
        )

        self.agent_paths = agent_paths
        self.agent_registry = AgentRuntimeRegistry()
        for agent_path in agent_paths:
            self.agent_registry.register_agent_from_path(path=agent_path)

        self.on_setup = on_setup

    def _get_pool(
        self,
        pool: Literal["prefork", "gevent", "threads", "solo"],
        workers: int,
    ) -> GeventPool | ThreadPoolExecutor | ProcessPoolExecutor | None:
        match pool:
            case "solo":
                return None
            case "gevent":
                return GeventPool(size=workers)
            case "threads":
                return ThreadPoolExecutor(max_workers=workers)
            case "prefork":
                broker_cls = type(self.broker)
                broker_kwargs = self.broker.get_init_kwargs()
                scheduler_cls = type(self.scheduler)
                session_repo_cls = type(self.session_repository)
                session_repo_kwargs: dict[str, Any] = (
                    self.session_repository.get_init_kwargs()
                )
                return ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initializer_worker,  # type: ignore
                    initargs=(  # type: ignore
                        scheduler_cls,
                        broker_cls,
                        broker_kwargs,
                        session_repo_cls,
                        session_repo_kwargs,
                        self.agent_paths,
                        self.on_setup,
                    ),
                )
            case _:
                raise ValueError(f"Invalid pool: {pool}")

    def _process_events(
        self,
        *,
        queues: Sequence[str],
        pool: GeventPool | ThreadPoolExecutor | ProcessPoolExecutor | None,
    ) -> None:
        while self.should_process_events:
            event = self.broker.get_event(queues=queues or ["default"])
            logger.info("Received event: %s", event.model_dump_json())
            if not pool:
                # No pool, run the event in the main thread
                handle_event_gracefully(
                    session_repository=self.session_repository,
                    scheduler=self.scheduler,
                    registry=self.agent_registry,
                    event=event,
                )
                continue

            if isinstance(pool, ProcessPoolExecutor):
                pool.submit(
                    _handle_process_pool_worker_event,
                    event=event,
                )
            elif isinstance(pool, ThreadPoolExecutor):
                pool.submit(
                    handle_event_gracefully,
                    session_repository=self.session_repository,
                    scheduler=self.scheduler,
                    registry=self.agent_registry,
                    event=event,
                )
            elif isinstance(pool, GeventPool):
                pool.spawn(
                    handle_event_gracefully,
                    session_repository=self.session_repository,
                    scheduler=self.scheduler,
                    registry=self.agent_registry,
                    event=event,
                )

    def start(
        self,
        *,
        queues: Sequence[str],
        pool: Literal["prefork", "gevent", "threads", "solo"],
        workers: int,
    ) -> None:
        if self.on_setup:
            self.on_setup()

        _render_header(
            workers=workers,
            pool=pool,
            queues=queues,
            broker=self.broker.get_display_name(),
            session_repository=self.session_repository.get_display_name(),
            registry=list(self.agent_registry.registry.values()),
        )

        runtime_pool = self._get_pool(pool=pool, workers=workers)
        logger.info("Listening for events...")
        try:
            self._process_events(queues=queues, pool=runtime_pool)
        except KeyboardInterrupt:
            console.print("\n[bold red]Shutting down...[/]")
            self._shutdown(pool=runtime_pool)

    def _shutdown(
        self,
        *,
        pool: GeventPool | ThreadPoolExecutor | ProcessPoolExecutor | None,
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
