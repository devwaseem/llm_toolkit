from pathlib import Path
from typing import Any, Callable, cast, override

from llm_toolkit.agentic.agents.base import Agent
from llm_toolkit.agentic.runtime.base import AgentRuntime
from llm_toolkit.agentic.session.base import AgentSession, AgentSessionReply
from llm_toolkit.llm.models import (
    LLM,
    LLMResponse,
    LLMStopReason,
    LLMToolCallRequest,
)
from llm_toolkit.tool import LLMTool, ToolKit, llm_tool


class SupervisorAgent(Agent):
    def __init__(
        self,
        *,
        llm: LLM,
        name: str,
        role: str,
        tools: list[LLMTool | ToolKit | Callable[..., str]],
        team: list[Agent],
        runtime: AgentRuntime,
        max_turns: int = 20,
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
            max_turns=max_turns,
            tools=_tools,
            runtime=runtime,
            additional_instructions=additional_instructions,
        )

        for agent in team:
            self._runtime.register_agent(agent=agent)

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

    @llm_tool(
        instructions=(
            "Always pass additional information to provide "
            "the context of task to the team member, "
            "since they start from clean slate."
        ),
        ignore_params={"from_session_id", "tool_request", "metadata"},
    )
    def delegate_to_agent(
        self,
        agent_id: str,
        task: str,
        additional_information: str,
        from_session_id: str,
        tool_request: LLMToolCallRequest,
        metadata: dict[str, Any] | None = None,
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
        agent = self.team[agent_id]
        session = self._runtime.session_repository.create_session(
            agent_id=agent.agent_id,
            reply_to=AgentSessionReply(
                session_id=from_session_id,
                tool_request=tool_request,
            ),
        )
        self._runtime.query_agent(
            query=(
                f"Task: {task}, "
                f"additional_information: {additional_information}"
            ),
            session_id=session.id,
            metadata=metadata,
        )
        return "Delegated task to team member"

    @override
    def run(
        self,
        *,
        session: AgentSession,
        metadata: dict[str, Any] | None = None,
        additional_context: str | None = None,
    ) -> LLMResponse:
        llm_response = super().run(
            session=session,
            metadata=metadata,
            additional_context=additional_context,
        )
        if llm_response.stop_reason == LLMStopReason.TOOL_USE and (
            llm_response.function_calls
        ):
            function_calls = []
            delegate_tool_name = LLMTool.from_callable(
                self.delegate_to_agent
            ).name
            for tool_call_request in llm_response.function_calls:
                if tool_call_request.name == delegate_tool_name:
                    self.delegate_to_agent(
                        **tool_call_request.arguments,
                        from_session_id=session.id,
                        tool_request=tool_call_request,
                        metadata=metadata,
                    )
                    session.add_tool_call_request(request=tool_call_request)
                    self._runtime.session_repository.save(session=session)
                    continue
                function_calls.append(tool_call_request)
            llm_response.function_calls = function_calls
            return llm_response

        return llm_response


class SubAgent(Agent):
    def get_system_message_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "subagent.txt"
