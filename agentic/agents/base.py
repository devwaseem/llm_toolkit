import re
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypedDict

from jinja2 import Environment
from tenacity import retry, retry_if_exception_type
from tenacity.wait import wait_exponential

from llm_toolkit.agentic.runner import AgentRunner
from llm_toolkit.agentic.session.base import AgentSession
from llm_toolkit.llm.errors import (
    LLMAPIConnectionError,
    LLMAPITimeoutError,
    LLMInternalServerError,
    LLMRateLimitedError,
)
from llm_toolkit.llm.models import (
    LLM,
    LLMResponse,
    LLMStopReason,
    LLMToolCallRequest,
    LLMToolRegistry,
    LLMTools,
)
from llm_toolkit.tool import LLMTool, ToolKit

if TYPE_CHECKING:
    pass


_jinja_env = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
)


class ToolContext(NamedTuple):
    tool_call_request: LLMToolCallRequest
    session: AgentSession
    runner: AgentRunner
    additional_context: str | None
    metadata: dict[str, Any] | None


class InternalToolMap(TypedDict):
    tool: LLMTool
    func: Callable[..., str]


class Agent:
    def __init__(
        self,
        *,
        llm: LLM,
        name: str,
        role: str,
        tools: list[LLMTool | ToolKit | Callable[..., str]],
        additional_instructions: str = "",
    ) -> None:
        self.llm = llm

        self.name = name
        self._validate_agent_name()

        self.role = role
        self.additional_instructions = additional_instructions

        self._tools = tools
        self._tool_registry = LLMToolRegistry()

        for tool in tools:
            self._tool_registry.add(tool)

        self._internal_tools: dict[str, InternalToolMap] = {}
        for callable_func in self.get_internal_tools():
            llm_tool = LLMTool.from_callable(callable_func)
            self._internal_tools[llm_tool.name] = {
                "tool": llm_tool,
                "func": callable_func,
            }
            self._tool_registry.add(llm_tool)

    def get_internal_tools(self) -> list[Callable[..., str]]:
        return []

    def get_llm_tools(self) -> list[LLMTool]:
        return self._tool_registry.get_tools()

    def get_system_message_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "agent.txt"

    def get_system_message_template(self) -> str:
        template_path = self.get_system_message_template_path()
        return template_path.read_text(encoding="utf-8")

    def get_system_message(
        self,
        *,
        session: AgentSession,
        additional_context: str | None = None,
    ) -> str:
        return _jinja_env.from_string(
            self.get_system_message_template()
        ).render(
            self.get_context_for_system_message(
                session=session,
                additional_context=additional_context,
            )
        )

    def get_context_for_system_message(
        self,
        *,
        session: AgentSession,  # noqa: ARG002
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "additional_context": additional_context,
            "additional_instructions": self.additional_instructions,
        }

    def _validate_agent_name(self) -> None:
        if len(self.name) == 0:
            raise ValueError("Invalid Agent name, should not be empty")
        if len(self.name) < 3:
            raise ValueError(
                "Invalid Agent name, should be at least 3 characters long"
            )
        if len(self.name) > 32:
            raise ValueError(
                "Invalid Agent name, should not be longer than 32 characters"
            )
        if re.match(r"^[a-zA-Z][a-zA-Z\s\-_]+$", self.name) is None:
            raise ValueError(
                "Invalid Agent name, "
                "only alphabets [a-zA-Z], underscore, hyphens and spaces "
                "are allowed. Name should start with a alphabet"
            )

    @cached_property
    def agent_id(self) -> str:
        return self.name.replace(" ", "_").lower()

    def get_short_tools_descriptions(self) -> list[tuple[str, str]]:
        return self._tool_registry.get_short_descriptions()

    @retry(
        retry=retry_if_exception_type(
            (
                LLMInternalServerError,
                LLMAPIConnectionError,
                LLMAPITimeoutError,
            )
        ),
        wait=wait_exponential(
            multiplier=2,
            max=10,
            exp_base=2,
        ),
    )
    @retry(
        retry=retry_if_exception_type((LLMRateLimitedError)),
        wait=wait_exponential(
            multiplier=2,
            max=60,
            exp_base=2,
        ),
    )
    def run(
        self,
        *,
        runner: AgentRunner,
        session: AgentSession,
        metadata: dict[str, Any] | None = None,
        additional_context: str | None = None,
    ) -> LLMResponse:
        system_message = self.get_system_message(
            session=session,
            additional_context=additional_context,
        )
        llm_response = self.llm.complete_chat(
            system_message=system_message,
            messages=[m.message for m in session.to_agent_messages()],
            tools=LLMTools(
                tools=self.get_llm_tools(),
                call_automatically=False,
                parallel_tool_calls=True,
            ),
            metadata=metadata,
        )
        if llm_response.stop_reason == LLMStopReason.TOOL_USE and (
            llm_response.function_calls
        ):
            function_calls = []
            for tool_call_request in llm_response.function_calls:
                if tool_call_request.name in self._internal_tools:
                    context = ToolContext(
                        tool_call_request=tool_call_request,
                        session=session,
                        runner=runner,
                        additional_context=additional_context,
                        metadata=metadata,
                    )
                    self._internal_tools[tool_call_request.name]["func"](
                        **tool_call_request.arguments,
                        context=context,
                    )
                    continue

                function_calls.append(tool_call_request)
            llm_response.function_calls = function_calls

        return llm_response
