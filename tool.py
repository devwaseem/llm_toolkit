import functools
import inspect
from typing import (
    Any,
    Callable,
    NamedTuple,
    ParamSpec,
    Set,
    TypeAliasType,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict

try:
    import docstring_parser
except ImportError as exc:
    raise ImportError(
        "Please install docstring_parser to use func2tool"
    ) from exc


Property = TypeAliasType("Property", "str | list[str] | dict[str, Property]")


class ToolParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    properties: dict[str, Property]
    required: list[str]


class ToolDef(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    instructions: str = ""

    def get_tool_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                self.description + f"Instructions: {self.instructions}"
            ),
            "parameters": self.parameters.model_dump(),
        }


class LLMTool(NamedTuple):
    name: str
    definition: ToolDef
    func: Callable[..., str] | None


P = ParamSpec("P")
R = TypeVar("R")


class CallableTool:
    name: str
    definition: ToolDef
    func: Callable[..., str]

    def __init__(self, func: Callable[..., str]) -> None:
        self.func = func
        if hasattr(self.func, "__is_llm_toolkit_tool__"):
            self.definition = cast(ToolDef, self.func.__llm_toolkit_tool_def__)  # type: ignore[attr-defined]
        else:
            self.definition = func2tool(self.func)

        if hasattr(self.func, "__llm_toolkit_tool_name__"):
            self.name = cast(str, self.func.__llm_toolkit_tool_name__)  # type: ignore[attr-defined]
            self.definition.name = self.name
        else:
            self.name = self.definition.name

    def to_llm_tool(self) -> LLMTool:
        return LLMTool(
            name=self.name,
            definition=self.definition,
            func=self.func,
        )


def llm_tool(
    *,
    name: str = "",
    instructions: str = "",
    ignore_params: Set[str] | None = None,
) -> Callable[[Callable[P, str]], Callable[P, str]]:
    ignore_params = ignore_params or set()
    ignore_params.add("self")

    def wrapper(func: Callable[P, str]) -> Callable[P, str]:
        @functools.wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> str:
            return func(*args, **kwargs)

        wrapped_func.__is_llm_toolkit_tool__ = True  # type: ignore[attr-defined]

        tool_def = func2tool(
            func,
            ignore_params=ignore_params,
        )

        tool_name = tool_def.name
        if name:
            # override the name of the tool
            tool_name = name

        tool_def.name = tool_name
        tool_def.instructions = instructions
        wrapped_func.__llm_toolkit_tool_name__ = tool_name  # type: ignore[attr-defined]
        wrapped_func.__llm_toolkit_tool_def__ = tool_def  # type: ignore[attr-defined]
        return wrapped_func

    return wrapper


def func2tool(
    func: Callable[..., Any],
    ignore_params: Set[str] | None = None,
    remove_prefix_from_name: str = "",
) -> ToolDef:
    """
    Converts a Python function into a JSON-serializable dictionary
        that describes the function's signature, including its name,
        description, and parameters.

        Args:
            func: The function to be converted.

        Returns:
            A dictionary representing the function's signature in JSON format.
    """

    ignore_params = ignore_params or set()

    try:
        signature = inspect.signature(func)
    except ValueError as exc:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {exc!s}"
        ) from exc

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    docstring = inspect.getdoc(func) or ""
    parsed_doc = docstring_parser.parse(docstring)

    # Build parameter descriptions from the parsed docstring
    param_descriptions = {p.arg_name: p.description for p in parsed_doc.params}

    parameters = ToolParameters(
        type="object",
        properties={},
        required=[],
    )

    for name, param in signature.parameters.items():
        if name in ignore_params:
            continue

        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as exc:
            raise KeyError(
                (
                    f"Unknown type annotation {param.annotation} "
                    f"for parameter {param.name}: {exc!s}"
                )
            ) from exc

        parameters.properties[name] = {
            "type": param_type,
            "description": param_descriptions.get(name, "") or "",
        }
        if param.default == inspect.Parameter.empty:
            parameters.required.append(name)

    function_name = func.__name__
    if remove_prefix_from_name:
        function_name = function_name.replace(remove_prefix_from_name, "")

    return ToolDef(
        name=function_name,
        description=(
            (parsed_doc.short_description or "")
            + (parsed_doc.long_description or "")
        ),
        instructions="",
        parameters=parameters,
    )


class ToolKit:
    def get_tools(self) -> list[LLMTool]:
        blacklist = {"get_tools"}
        toolkit_name = self.__class__.__name__
        toolkit_name = toolkit_name.removesuffix("ToolKit").removesuffix(
            "Toolkit"
        )
        tools: list[LLMTool] = []
        for attr in dir(self.__class__):
            if (
                callable(getattr(self.__class__, attr))
                and not attr.startswith("_")
                and attr not in blacklist
            ):
                cls_tool = getattr(self.__class__, attr)
                tool_def: ToolDef
                is_llm_toolkit_tool = hasattr(
                    cls_tool, "__is_llm_toolkit_tool__"
                )
                if is_llm_toolkit_tool:
                    tool_def = cls_tool.__llm_toolkit_tool_def__  # type: ignore[attr-defined]
                else:
                    tool_def = func2tool(
                        cls_tool,
                        ignore_params={"self"},
                    )

                if toolkit_name not in tool_def.name:
                    new_name = toolkit_name + "__" + tool_def.name
                    tool_def.name = new_name
                    if is_llm_toolkit_tool:
                        cls_tool.__llm_toolkit_tool_name__ = new_name  # type: ignore[attr-defined]

                tools.append(
                    LLMTool(
                        name=tool_def.name,
                        definition=tool_def,
                        func=getattr(self, attr),
                    )
                )

        return tools
