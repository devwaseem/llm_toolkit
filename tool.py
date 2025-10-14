import functools
import inspect
import types
from enum import StrEnum
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    ParamSpec,
    Set,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
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
    short_description: str
    long_description: str
    parameters: ToolParameters
    instructions: str = ""

    @property
    def description(self) -> str:
        return f"{self.short_description}\n{self.long_description}"

    def get_tool_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                self.description + f"Instructions: {self.instructions}"
            ),
            "parameters": self.parameters.model_dump(),
        }


class LLMTool:
    definition: ToolDef
    func: Callable[..., str] | None

    def __init__(
        self,
        *,
        definition: ToolDef,
        display_name: str = "",
        func: Callable[..., str] | None,
    ) -> None:
        self.definition = definition
        self.func = func
        self.display_name = display_name

    @staticmethod
    def from_callable(func: Callable[..., str]) -> "LLMTool":
        if inspect.isclass(func):
            raise TypeError("func must be a function, not a class")

        if not callable(func):
            raise TypeError("func must be a callable")

        is_llm_tool = hasattr(func, "__is_llm_tool__")
        if is_llm_tool:
            definition = cast(ToolDef, func.__llm_tool_def__)  # type: ignore[attr-defined]
        else:
            definition = func2tool(
                func,
                ignore_params={
                    "self",
                    "metadata",
                },
            )

        if is_llm_tool and hasattr(func, "__llm_tool_name__"):
            definition.name = cast(str, func.__llm_tool_name__)  # type: ignore[attr-defined]

        display_name = ""
        if is_llm_tool and hasattr(func, "__llm_tool_display_name__"):
            display_name = cast(str, func.__llm_tool_display_name__)  # type: ignore[attr-defined]

        return LLMTool(
            definition=definition,
            display_name=display_name,
            func=func,
        )

    @property
    def name(self) -> str:
        return self.definition.name

    @name.setter
    def name(self, name: str) -> None:
        self.definition.name = name

    @property
    def is_metadata_requested(self) -> bool:
        if not self.func:
            return False
        return "metadata" in inspect.signature(self.func).parameters

    def __repr__(self) -> str:
        return (
            f"LLMTool(name={self.name}, definition={self.definition}, "
            f"func={self.func.__name__ if self.func else None})"
        )


P = ParamSpec("P")
R = TypeVar("R")


def llm_tool(
    *,
    name: str = "",
    instructions: str = "",
    display_name: str = "",
    ignore_params: Set[str] | None = None,
) -> Callable[[Callable[P, str]], Callable[P, str]]:
    ignore_params = ignore_params or set()
    ignore_params.add("self")
    ignore_params.add("metadata")

    def wrapper(func: Callable[P, str]) -> Callable[P, str]:
        @functools.wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> str:
            return func(*args, **kwargs)

        wrapped_func.__is_llm_tool__ = True  # type: ignore[attr-defined]

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
        wrapped_func.__llm_tool_name__ = tool_name  # type: ignore[attr-defined]
        wrapped_func.__llm_tool_def__ = tool_def  # type: ignore[attr-defined]
        wrapped_func.__llm_tool_display_name__ = display_name  # type: ignore[attr-defined]
        return wrapped_func

    return wrapper


class _ResolvedAnnotation(NamedTuple):
    param_type: str
    schema_props: dict[str, Any]
    is_optional: bool


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

        param_type, schema_props, is_optional = _resolve_annotation(
            param.annotation, type_map
        )

        properties: dict[str, Property] = {
            "type": param_type,
            "description": param_descriptions.get(name, "") or "",
        }
        properties.update(schema_props)

        # if param.default:
        #     properties["default"] = param.default

        parameters.properties[name] = properties

        if not is_optional and param.default == inspect.Parameter.empty:
            parameters.required.append(name)

    function_name = func.__name__
    if remove_prefix_from_name:
        function_name = function_name.replace(remove_prefix_from_name, "")

    return ToolDef(
        name=function_name,
        short_description=parsed_doc.short_description or "",
        long_description=parsed_doc.long_description or "",
        instructions="",
        parameters=parameters,
    )


def _resolve_annotation(
    annotation: Any, type_map: dict[type, str]
) -> _ResolvedAnnotation:
    param_type = "string"
    schema_props: dict[str, Any] = {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    is_optional = False
    if origin is types.UnionType or origin is Union:
        is_optional = type(None) in args
        non_none_type = [a for a in args if a is not type(None)]
        return _resolve_annotation(non_none_type[0], type_map)

    if origin is Literal:
        param_type = "string"
        schema_props["enum"] = list(args)
        first_type = type(args[0])

        if first_type not in (str, int, float):
            raise TypeError(
                "Only Literal with str, int, float types are supported"
            )

        count = 1
        for arg in args[1:]:
            if type(arg) is first_type:
                count += 1

        if count != len(args):
            raise TypeError("Literal should contain same type")

        if isinstance(first_type, (int, float)):
            param_type = "integer"  # type: ignore

        return _ResolvedAnnotation(
            param_type=param_type,
            schema_props=schema_props,
            is_optional=is_optional,
        )

    if inspect.isclass(annotation) and issubclass(annotation, StrEnum):
        schema_props["enum"] = [m.value for m in annotation]
        return _ResolvedAnnotation(
            param_type="string",
            schema_props=schema_props,
            is_optional=is_optional,
        )

    param_type = type_map.get(annotation, "string")
    return _ResolvedAnnotation(
        param_type=param_type,
        schema_props=schema_props,
        is_optional=is_optional,
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
                func = getattr(self, attr)
                tool = LLMTool.from_callable(func)

                if toolkit_name not in tool.name:
                    tool.name = toolkit_name + "__" + tool.name

                tools.append(tool)

        return tools
