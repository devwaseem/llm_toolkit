import inspect
from typing import Any, Callable, Type, get_type_hints

import docstring_parser


def func2tool(func: Callable[..., ...]) -> dict[str, Any]:
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring = inspect.getdoc(func) or ""
    parsed_doc = docstring_parser.parse(docstring)

    # Build parameter descriptions from the parsed docstring
    param_descriptions = {p.arg_name: p.description for p in parsed_doc.params}

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    for name, param in signature.parameters.items():
        param_type: Type[Any] = type_hints.get(name, str)
        json_type = "string"  # default fallback

        if param_type is int:  # type: ignore
            json_type = "integer"
        elif param_type is float:  # type: ignore
            json_type = "number"
        elif param_type is bool:  # type: ignore
            json_type = "boolean"
        elif param_type is list:  # type: ignore
            json_type = "array"
        elif param_type is dict:  # type: ignore
            json_type = "object"

        parameters["properties"][name] = {
            "type": json_type,
            "description": param_descriptions.get(name, ""),
        }
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "name": func.__name__,
        "description": (
            (parsed_doc.short_description or "")
            + (parsed_doc.long_description or "")
        ),
        "parameters": parameters,
    }
