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
    example_file = Path(__file__).parent / "examples" / "abbreviated.txt"
    return example_file.read_text()


class AbbreviatedKeyLLMSchemaGenerator(LLMSchemaGenerator):
    def __init__(
        self,
        *,
        schema: Type[LLMSchemaModelTypeVar],
        encoded: bool = True,
    ) -> None:
        super().__init__(schema=schema, encoded=encoded)
        self.key_cache: dict[str, str] = {}

    @override
    def build_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for attr in dir(self.schema):
            if attr.startswith("_"):
                continue

            attr_value = getattr(self.schema, attr)

            if isinstance(attr_value, Field):
                key = self._encode_key(key=attr)
                schema[key] = self._get_schema_for_field(
                    name=attr, field_value=attr_value
                )

            elif isinstance(attr_value, LLMSchemaModel):
                key = self._encode_key(key=attr)
                schema[key] = self._get_schema_for_model(
                    name=attr,
                    schema=attr_value,
                )

            elif isinstance(attr_value, ListField):
                key = self._encode_key(key=attr)
                if issubclass(attr_value.cls, Field):
                    field_instance = attr_value.cls(**attr_value.kwargs or {})
                    schema[key] = [
                        self._get_schema_for_field(
                            name=attr, field_value=field_instance
                        )
                    ]
                elif issubclass(attr_value.cls, LLMSchemaModel):
                    model_instance = attr_value.cls(**attr_value.kwargs or {})
                    schema[key] = [
                        self._get_schema_for_model(
                            name=attr,
                            schema=model_instance,
                        )
                    ]
                else:
                    raise NotImplementedError

        return schema

    def _encode_key(self, *, key: str) -> str:
        if not self.encoded:
            return key

        encoded_key = self.generate_key(original_key=key)
        self.key_cache[encoded_key] = key
        return encoded_key

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

    def generate_key(self, *, original_key: str) -> str:
        int_key = 0
        for char in original_key:
            int_key += ord(char)
        return str(int_key % 99) + "".join(
            [i[0] for i in original_key.split("_")]
        )

    def _get_schema_for_field(
        self,
        *,
        name: str,
        field_value: Field[Any],
    ) -> dict[str, Any]:
        schema_obj = {
            "type": field_value.value_type,
            "name": name,
        }
        if field_value.description:
            schema_obj["description"] = field_value.description
        if field_value.extra_args:
            schema_obj.update(field_value.extra_args)

        return schema_obj

    def _get_schema_for_model(
        self,
        *,
        name: str,
        schema: LLMSchemaModel,
    ) -> dict[str, Any]:
        child_schema = {
            "name": name,
            "schema": self.__class__(
                schema=schema.__class__,
                encoded=self.encoded,
            ).build_schema(),
        }
        if schema.description:
            child_schema["description"] = schema.description

        return child_schema

    def get_example(self) -> str:
        return _get_example()
