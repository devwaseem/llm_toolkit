from functools import cache
from pathlib import Path
from typing import Any, override

from llm_toolkit.schema_generator.exceptions import (
    InvalidSchemaError,
)
from llm_toolkit.schema_generator.models import (
    Field,
    ListField,
    LLMSchemaGenerator,
    LLMSchemaModel,
)


@cache
def _get_example() -> str:
    example_file = Path(__file__).parent / "examples" / "definition_driven.txt"
    return example_file.read_text()


class DefinitionDrivenLLMSchemaGenerator(LLMSchemaGenerator):
    def __init__(self, *, schema: LLMSchemaModel, encode: bool = True) -> None:
        self.schema = schema
        self.should_encode = encode
        self.key_cache: dict[str, str] = {}
        self.definitions: dict[str, Any] = {}
        self._collect_schema_definitions(schema=self.schema)
        self.fields = self._collect_schema_fields(schema=self.schema)

    def build_schema(self) -> dict[str, Any]:
        return {
            "strict": True,
            "definitions": self.definitions,
            "fields": self.fields,
        }

    def _collect_schema_definitions(self, *, schema: LLMSchemaModel) -> None:
        for attr in dir(schema):
            if attr.startswith("_"):
                continue

            field = getattr(schema, attr)
            if isinstance(field, LLMSchemaModel):
                model_name = type(field).__name__
                if model_name not in self.definitions:
                    definition = {
                        "fields": self._collect_schema_fields(schema=field)
                    }
                    self.definitions[model_name] = definition
                    self._collect_schema_definitions(schema=field)
            elif isinstance(field, ListField):
                model_name = field.cls.__name__
                if model_name not in self.definitions:
                    field_instance = field.cls(**field.kwargs or {})
                    definition = {
                        model_name: {
                            "fields": self._collect_schema_fields(
                                schema=field_instance
                            ),
                        }
                    }
                    self.definitions[model_name] = definition
                    self._collect_schema_definitions(schema=field_instance)

    def _collect_schema_fields(
        self, *, schema: LLMSchemaModel
    ) -> dict[str, Any]:
        fields = {}
        for attr in dir(schema):
            if attr.startswith("_"):
                continue

            field = getattr(schema, attr)

            schema_obj: dict[str, Any] = {}
            if isinstance(field, Field):
                if self.should_encode:
                    schema_obj["name"] = attr
                if field.description:
                    schema_obj["description"] = field.description
                if field.extra_args:
                    schema_obj.update(field.extra_args)

                schema_obj["type"] = field.value_type

            elif isinstance(field, LLMSchemaModel):
                model_name = type(field).__name__
                if self.should_encode:
                    schema_obj["name"] = attr

                if field.description:
                    schema_obj["description"] = field.description

                schema_obj["type"] = "object"
                schema_obj["object_type"] = model_name

            elif isinstance(field, ListField):
                model_name = field.cls.__name__
                field_instance = field.cls(**field.kwargs or {})

                if self.should_encode:
                    schema_obj["name"] = attr

                if field_instance.description:
                    schema_obj["description"] = field_instance.description

                schema_obj["type"] = "list"
                schema_obj["list_object"] = model_name

            if schema_obj:
                encoded_key = (
                    self._generate_key(original_key=attr)
                    if self.should_encode
                    else attr
                )
                fields[encoded_key] = schema_obj
                self.key_cache[encoded_key] = attr

        return fields

    @override
    def is_encoded(self) -> bool:
        return self.should_encode

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
                decoded_schema[decoded_key] = [
                    self.decode_json(data=item) for item in value
                ]
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
