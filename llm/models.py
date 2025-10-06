import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum
from typing import (
    Any,
    Callable,
    NamedTuple,
    Sequence,
    TypeVar,
)

from pydantic import BaseModel

from llm_toolkit.models import LLMFileData
from llm_toolkit.tool import LLMTool, ToolDef, ToolKit

logger = logging.getLogger(__name__)


class LLMOutputMode(StrEnum):
    TEXT = "TEXT"
    JSON = "JSON"


class LLMMessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_OUTPUT = "TOOL_OUTPUT"


class LLMInputFile(BaseModel):
    text: str
    file: LLMFileData


class LLMToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMToolCallResponse(BaseModel):
    tool_call: LLMToolCallRequest
    output: str


class LLMInputMessage(BaseModel):
    role: LLMMessageRole
    content: str | LLMInputFile | LLMToolCallRequest | LLMToolCallResponse

    @staticmethod
    def from_human(content: str | LLMInputFile) -> "LLMInputMessage":
        return LLMInputMessage(
            role=LLMMessageRole.USER,
            content=content,
        )

    @staticmethod
    def from_ai(content: str | LLMInputFile) -> "LLMInputMessage":
        return LLMInputMessage(
            role=LLMMessageRole.ASSISTANT,
            content=content,
        )

    @staticmethod
    def from_tool_call_request(
        tool_call: LLMToolCallRequest,
    ) -> "LLMInputMessage":
        return LLMInputMessage(
            role=LLMMessageRole.TOOL_CALL,
            content=tool_call,
        )

    @staticmethod
    def from_tool_response(
        response: LLMToolCallResponse,
    ) -> "LLMInputMessage":
        return LLMInputMessage(
            role=LLMMessageRole.TOOL_OUTPUT,
            content=response,
        )


class LLMStopReason(StrEnum):
    MAX_TOKENS = "MAX_TOKENS"
    END_TURN = "END_TURN"
    TOOL_USE = "TOOL_USE"
    STOP_SEQUENCE = "STOP_SEQUENCE"


class LLMResponse(BaseModel):
    llm_model: str
    answer: str
    prompt_tokens_used: int
    completion_tokens_used: int
    cost: float
    function_calls: list[LLMToolCallRequest] | None = None
    stop_reason: LLMStopReason = LLMStopReason.END_TURN


class LLMPriceCalculator(NamedTuple):
    tokens: int
    input_tokens: float
    output_tokens: float

    def cost_per_input_token(self) -> Decimal:
        return Decimal(self.input_tokens) / Decimal(max(1, self.tokens))

    def cost_per_output_token(self) -> Decimal:
        return Decimal(self.output_tokens) / Decimal(max(1, self.tokens))

    def calculate_price(
        self, input_tokens: int, output_tokens: int
    ) -> Decimal:
        cost_of_input_tokens = self.cost_per_input_token() * input_tokens
        cost_of_output_tokens = self.cost_per_output_token() * output_tokens
        return cost_of_input_tokens + cost_of_output_tokens


class LLMTokenBudget:
    llm_max_token: int
    max_tokens_for_context: int
    _max_tokens_for_output: int | None

    def __init__(
        self,
        llm_max_token: int,
        max_tokens_for_input: int,
        max_tokens_for_output: int | None = None,
    ) -> None:
        self.llm_max_token = llm_max_token
        self._max_tokens_for_output = max_tokens_for_output
        if max_tokens_for_input > llm_max_token:
            raise ValueError(
                "max_tokens_for_context cannot be larger than llm_max_token"
            )
        self.max_tokens_for_context = max_tokens_for_input

    @property
    def max_tokens_for_output(self) -> int:
        if not self._max_tokens_for_output:
            return self.llm_max_token - self.max_tokens_for_context
        return self._max_tokens_for_output


class LLMTools(NamedTuple):
    tools: Sequence[Callable[..., str] | ToolKit | ToolDef | LLMTool]
    call_automatically: bool = True
    parallel_tool_calls: bool = True


class ToolNotCallableError(Exception):
    """Tool is not registered to be callable"""


class LLMToolRegistry:
    def __init__(self) -> None:
        self.registry: dict[str, LLMTool] = {}

    def add(
        self, tool: Callable[..., str] | ToolKit | ToolDef | LLMTool
    ) -> None:
        if isinstance(tool, LLMTool):
            self.registry[tool.name] = tool

        elif isinstance(tool, ToolKit):
            for t in tool.get_tools():
                self.registry[t.name] = t

        elif isinstance(tool, ToolDef):
            llm_tool = LLMTool(definition=tool, func=None)
            self.registry[llm_tool.name] = llm_tool

        elif callable(tool):
            llm_tool = LLMTool.from_callable(tool)
            self.registry[llm_tool.name] = llm_tool

        else:
            raise TypeError(f"Unsupported tool type: {tool}")

    def process_tool_calls(
        self,
        messages: list[LLMInputMessage],
        tool_calls: list[LLMToolCallRequest],
        metadata: dict[str, Any],
    ) -> list[LLMInputMessage]:
        if not tool_calls:
            return messages

        messages = messages.copy()
        for tool_call in tool_calls:
            messages.append(LLMInputMessage.from_tool_call_request(tool_call))
            logger.info("Calling tool: %s", tool_call.name)
            try:
                output = self.execute(
                    tool_call=tool_call,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to execute tool: %s",
                    tool_call.name,
                    exc_info=exc,
                    stack_info=False,
                )
                output = (
                    f"Failed to execute tool: {tool_call.name}. "
                    f"Reason (Internal Use): {exc!s}"
                )
            logger.debug("Tool output: [%s] -> %s", tool_call.name, output)
            messages.append(
                LLMInputMessage.from_tool_response(
                    LLMToolCallResponse(
                        tool_call=tool_call,
                        output=str(output),
                    )
                )
            )

        return messages

    def execute(
        self,
        *,
        tool_call: LLMToolCallRequest,
        metadata: dict[str, Any],
    ) -> str:
        if tool_call.name not in self.registry:
            raise ValueError(f"Unknown tool: {tool_call.name}")

        tool = self.registry[tool_call.name]
        if not tool.func:
            raise ToolNotCallableError(
                f"Tool {tool.name} is not registered to be callable"
            )

        tool_call_arguments = tool_call.arguments
        if tool.is_metadata_requested:
            tool_call_arguments["metadata"] = metadata

        return tool.func(**tool_call_arguments)

    @property
    def definitions(self) -> list[ToolDef]:
        return [tool.definition for tool in self.registry.values()]

    def get_short_descriptions(self) -> list[tuple[str, str]]:
        return [(i.name, i.description) for i in self.definitions]

    def get_tools(self) -> list[LLMTool]:
        return list(self.registry.values())

    def __str__(self) -> str:
        return "\n\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.definitions]
        )


class LLM(ABC):
    @abstractmethod
    def get_api_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        temperature: float = 0,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
        tools: LLMTools | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class StructuredOutputLLM(ABC):
    @abstractmethod
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
        tools: LLMTools | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[PydanticModel | None, LLMResponse]:
        raise NotImplementedError
