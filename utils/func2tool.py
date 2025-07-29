import inspect
from typing import Any, Callable, Set, TypedDict, get_type_hints

try:
    import docstring_parser  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Please install docstring_parser to use func2tool"
    ) from exc


class ToolParamProperty(TypedDict):
    type: str
    description: str


class ToolParameter(TypedDict):
    type: str
    properties: dict[str, ToolParamProperty]
    required: list[str]


class Tool(TypedDict):
    name: str
    description: str
    parameters: ToolParameter


def func2tool(
    func: Callable[..., Any],
    ignore_params: Set[str] | None = None,
) -> Tool:
    ignore_params = ignore_params or set()

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring = inspect.getdoc(func) or ""
    parsed_doc = docstring_parser.parse(docstring)

    # Build parameter descriptions from the parsed docstring
    param_descriptions = {p.arg_name: p.description for p in parsed_doc.params}

    parameters: ToolParameter = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    for name, param in signature.parameters.items():
        if name in ignore_params:
            continue

        param_type: type[Any] = type_hints.get(name, str)
        json_type = "string"  # default fallback

        if param_type is int:
            json_type = "integer"
        elif param_type is float:
            json_type = "number"
        elif param_type is bool:
            json_type = "boolean"
        elif param_type is list:
            json_type = "array"
        elif param_type is dict:
            json_type = "object"

        parameters["properties"][name] = {
            "type": json_type,
            "description": param_descriptions.get(name, ""),
        }
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return Tool(
        name=func.__name__,
        description=(
            (parsed_doc.short_description or "")
            + (parsed_doc.long_description or "")
        ),
        parameters=parameters,
    )
