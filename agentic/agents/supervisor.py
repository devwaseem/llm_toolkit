from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast, override

from llm_toolkit.agentic.agents.base import Agent, ToolContext
from llm_toolkit.agentic.session.base import AgentSession, AgentSessionReply
from llm_toolkit.llm.models import (
    LLM,
    LLMToolCallResponse,
)
from llm_toolkit.tool import LLMTool, ToolKit, llm_tool

if TYPE_CHECKING:
    pass


class SupervisorAgent(Agent):
    def __init__(
        self,
        *,
        llm: LLM,
        name: str,
        role: str,
        tools: list[LLMTool | ToolKit | Callable[..., str]],
        team: list[Agent],
        additional_instructions: str = "",
    ) -> None:
        self.team = {t.agent_id: t for t in team}

        delegate_to_agent_tool = LLMTool.from_callable(self.delegate_to_agent)
        agent_id_prop = cast(
            dict[str, Any],
            delegate_to_agent_tool.definition.parameters.properties[
                "agent_id"
            ],
        )
        agent_id_prop["enum"] = list(self.team.keys())

        _tools = [
            *tools,
            delegate_to_agent_tool,
        ]

        super().__init__(
            llm=llm,
            name=name,
            role=role,
            tools=_tools,
            additional_instructions=additional_instructions,
        )

    @override
    def get_system_message_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "supervisor.txt"

    @override
    def get_context_for_system_message(
        self,
        *,
        session: AgentSession,
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        context = super().get_context_for_system_message(
            session=session,
            additional_context=additional_context,
        )
        context["team"] = [
            {
                "id": t.agent_id,
                "name": t.name,
                "role": t.role,
                "tools": t.get_short_tools_descriptions(),
            }
            for t in self.team.values()
        ]
        return context

    @override
    def get_internal_tools(self) -> list[Callable[..., str]]:
        return [
            *super().get_internal_tools(),
            self.delegate_to_agent,
        ]

    @llm_tool(
        instructions=(
            "Always pass additional information to provide "
            "the context of task to the team member, "
            "since they start from clean slate."
        ),
        ignore_params={
            "context",
        },
    )
    def delegate_to_agent(
        self,
        agent_id: str,
        task: str,
        additional_information: str,
        context: ToolContext,
    ) -> str:
        """
        Delegate a task to a team member

        Args:
            agent_id (str): id of the agent to delegate the task to
            task (str): Task for the agent to complete and respond back
            additional_information (str): additional context/information since
            the team member doesn't know the context of the task and why it
            was initiated.
        """
        from_session = context.session
        runner = context.runner

        with runner.session_repository.select_for_update(
            session_id=from_session.id
        ) as s:
            s.add_tool_call_request(request=context.tool_call_request)

        agent = self.team[agent_id]
        delegate_session = runner.session_repository.create_session(
            agent_id=agent.agent_id,
            reply_to=AgentSessionReply(
                session_id=from_session.id,
                tool_request=context.tool_call_request,
            ),
        )
        runner.query_agent(
            query=(
                f"Task: {task}, "
                f"additional_information: {additional_information}"
            ),
            session_id=delegate_session.id,
            metadata=context.metadata,
        )
        return "Delegated task to team member"

    @llm_tool(
        display_name="Thinking",
        instructions=(
            "Use this tool to set the agent's thinking."
            "This will be used for the user to know what the agent is thinking."
            "Do not include any internal knowledge here, as this is user facing."
        ),
        ignore_params={"context"},
    )
    def set_internal_thoughts(
        self,
        thoughts: str,  # noqa
        context: ToolContext,
    ) -> str:
        """Set the thoughts of the agent"""
        runner = context.runner
        session = context.session
        tool_call_request = context.tool_call_request

        with runner.session_repository.select_for_update(
            session_id=session.id
        ) as s:
            s.current_thoughts = tool_call_request.arguments["thoughts"]
            s.add_tool_call_request(request=tool_call_request)
            s.add_tool_call_response(
                from_session_id=session.id,
                response=LLMToolCallResponse(
                    tool_call=tool_call_request,
                    output="OK",
                ),
            )

        runner.scheduler.schedule_agent_event(
            session_id=session.id,
            addtional_context=context.additional_context,
            metadata=context.metadata,
        )

        return ""


class SubAgent(Agent):
    def get_system_message_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "subagent.txt"
