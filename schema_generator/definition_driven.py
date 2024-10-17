import inspect
from functools import cache
from pathlib import Path
from typing import Any, Type, override

from llm_toolkit.schema_generator.exceptions import (
    InvalidSchemaError,
)
from llm_toolkit.schema_generator.models import (
    Field,
    ListField,
    LLMSchemaGenerator,
    LLMSchemaModel,
    LLMSchemaModelTypeVar,
)


@cache
def _get_example() -> str:
    example_file = Path(__file__).parent / "examples" / "definition_driven.txt"
    return example_file.read_text()


class DefinitionDrivenLLMSchemaGenerator(LLMSchemaGenerator):
    def __init__(
        self,
        *,
        schema_cls: Type[LLMSchemaModelTypeVar],
        encoded: bool = True,
    ) -> None:
        super().__init__(schema=schema_cls, encoded=encoded)
        self.key_cache: dict[str, str] = {}
        self.definitions: dict[str, Any] = {}
        self.fields = self._collect_schema_fields(schema=schema_cls)
        self._collect_schema_definitions(schema=schema_cls)

        if not inspect.isclass(schema_cls):
            raise ValueError("schema should be a class")

    def build_schema(self) -> dict[str, Any]:
        return {
            "strict": True,
            "definitions": self.definitions,
            "fields": self.fields,
        }

    def _collect_schema_definitions(
        self, *, schema: Type[LLMSchemaModelTypeVar]
    ) -> None:
        for attr in dir(schema):
            if attr.startswith("_"):
                continue

            field = getattr(schema, attr)
            if isinstance(field, LLMSchemaModel):
                model_name = type(field).__name__
                if model_name not in self.definitions:
                    self.definitions[model_name] = self._collect_schema_fields(
                        schema=field.__class__
                    )
                    self._collect_schema_definitions(schema=field.__class__)
            elif isinstance(field, ListField):
                model_name = field.cls.__name__
                if model_name not in self.definitions:
                    self.definitions[model_name] = self._collect_schema_fields(
                        schema=field.cls  # type: ignore
                    )

    def _collect_schema_fields(  # noqa
        self, *, schema: Type[LLMSchemaModelTypeVar]
    ) -> dict[str, Any]:
        fields = {}
        for attr in dir(schema):
            if attr.startswith("_"):
                continue

            field = getattr(schema, attr)

            schema_obj: dict[str, Any] = {}
            if isinstance(field, Field):
                if self.encoded:
                    schema_obj["name"] = attr
                if field.description:
                    schema_obj["description"] = field.description
                if field.extra_args:
                    schema_obj.update(field.extra_args)

                schema_obj["type"] = field.value_type

            elif isinstance(field, LLMSchemaModel):
                model_name = type(field).__name__
                if self.encoded:
                    schema_obj["name"] = attr

                if field.description:
                    schema_obj["description"] = field.description

                schema_obj["type"] = "object"
                schema_obj["object_type"] = model_name

            elif isinstance(field, ListField):
                if self.encoded:
                    schema_obj["name"] = attr
                field_instance = field.cls(**field.kwargs or {})

                if field_instance.description:
                    schema_obj["description"] = field_instance.description

                schema_obj["type"] = "list"

                if isinstance(field_instance, LLMSchemaModel):
                    schema_obj["list_object"] = field.cls.__name__
                elif isinstance(field_instance, Field):
                    schema_obj["list_object"] = field_instance.value_type

            if schema_obj:
                encoded_key = (
                    self._generate_key(original_key=attr)
                    if self.encoded
                    else attr
                )
                fields[encoded_key] = schema_obj
                self.key_cache[encoded_key] = attr

        return fields

    @override
    def decode_json(self, *, data: dict[str, Any]) -> dict[str, Any]:
        decoded_schema: dict[str, Any] = {}
        for key, value in data.items():
            if key not in self.key_cache:
                raise InvalidSchemaError(
                    f"Key {key} not encoded using this instance"
                )

            decoded_key = self.key_cache[key]
            if isinstance(value, dict):
                decoded_schema[decoded_key] = self.decode_json(data=value)
            elif isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict):
                        items.append(self.decode_json(data=item))
                    else:
                        items.append(item)

                decoded_schema[decoded_key] = items
            else:
                decoded_schema[decoded_key] = value
        return decoded_schema

    def _generate_key(self, *, original_key: str) -> str:
        int_key = 0
        for char in original_key:
            int_key += ord(char)
        return str(int_key % 99) + "".join(
            [i[0] for i in original_key.split("_")]
        )

    def get_example(self) -> str:
        return _get_example()
