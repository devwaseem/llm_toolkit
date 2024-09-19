from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias, override

LLMSchema: TypeAlias = dict[str, Any]


class LLMSchemaCodecInterface(ABC):
    def __init__(self) -> None:
        self.key_cache: dict[str, str] = {}

    def encode(self, *, schema: LLMSchema) -> LLMSchema:
        encoded_schema: LLMSchema = {}
        for key, value in schema.items():
            encoded_key = self.generate_key(original_key=key)
            self.key_cache[encoded_key] = key
            if isinstance(value, dict):
                encoded_schema[encoded_key] = self.encode(schema=value)
            elif isinstance(value, list):
                encoded_schema[encoded_key] = [
                    self.encode(schema=item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                encoded_schema[encoded_key] = value
        return encoded_schema

    def decode(self, *, schema: LLMSchema) -> LLMSchema:
        decoded_schema: LLMSchema = {}
        for key, value in schema.items():
            if key not in self.key_cache:
                raise CompactLLMSchemaCodecNotEncodedError(
                    f"Key {key} not encoded using this instance"
                )

            decoded_key = self.key_cache[key]
            if isinstance(value, dict):
                decoded_schema[decoded_key] = self.decode(schema=value)
            elif isinstance(value, list):
                decoded_schema[decoded_key] = [
                    self.decode(schema=item) for item in value
                ]
            else:
                decoded_schema[decoded_key] = value
        return decoded_schema

    @abstractmethod
    def generate_key(self, *, original_key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_llm_example_prompt(self) -> str:
        raise NotImplementedError


class DummyLLMSchemaCodec(LLMSchemaCodecInterface):
    @override
    def encode(self, *, schema: LLMSchema) -> LLMSchema:
        return schema

    @override
    def decode(self, *, schema: LLMSchema) -> LLMSchema:
        return schema

    @override
    def generate_key(self, *, original_key: str) -> str:
        return original_key

    @override
    def get_llm_example_prompt(self) -> str:
        return ""


class CompactLLMSchemaCodecNotEncodedError(Exception):
    ...


class CompactLLMSchemaCodec(LLMSchemaCodecInterface):
    def __init__(self, exclude_keys: set[str]) -> None:
        self.exclude_keys = exclude_keys
        super().__init__()

    @override
    def generate_key(self, *, original_key: str) -> str:
        if original_key in self.exclude_keys:
            return original_key

        int_key = 0
        for char in original_key:
            int_key += ord(char)
        return str(int_key % 99) + "".join(
            [i[0] for i in original_key.split("_")]
        )

    @override
    def get_llm_example_prompt(self) -> str:
        prompt_file = (
            Path(__file__).parent / "example_prompts" / "compact_codec.txt"
        )
        return prompt_file.read_text()
