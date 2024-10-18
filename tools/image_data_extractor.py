import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar, cast

import structlog

from llm_toolkit.llm.models import (
    LLM,
    LLMInputMessage,
    LLMMessageBuilderInterface,
    LLMMessageRole,
    LLMOutputMode,
    LLMResponse,
)
from llm_toolkit.schema_generator.models import (
    LLMSchemaGenerator,
    LLMSchemaModel,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=LLMSchemaModel)


class LLMExtractedImageData(NamedTuple, Generic[T]):
    schema: T
    llm_response: LLMResponse


def extract_schema_data_from_image(
    *,
    llm: LLM,
    llm_message_builder: LLMMessageBuilderInterface,
    system_message: str,
    image_file: Path,
    schema_generator: LLMSchemaGenerator[T],
    pre_image_llm_messages: list[LLMInputMessage] | None = None,
    post_image_llm_messages: list[LLMInputMessage] | None = None,
) -> LLMExtractedImageData[T]:
    schema_dict = schema_generator.build_schema()

    llm_messages = pre_image_llm_messages or []
    with image_file.open("rb") as image_fd:
        base64_image = base64.b64encode(image_fd.read()).decode("utf-8")
        mime_type = mimetypes.guess_type(url=str(image_file))[0] or "image/*"
        schema_str = json.dumps(schema_dict)
        logger.debug(
            "generated schema",
            schema=schema_str,
        )
        llm_messages.append(
            (
                llm_message_builder.add_base64_image(
                    mime_type=mime_type,
                    content=f"data:image/png;base64,{base64_image}",
                )
                .add_text(text=schema_str)
                .build_message(role=LLMMessageRole.USER)
            )
        )
    if post_image_llm_messages:
        llm_messages.extend(post_image_llm_messages)

    llm_response = llm.complete_chat(
        system_message=(system_message + schema_generator.get_example()),
        messages=llm_messages,
        output_mode=LLMOutputMode.JSON,
    )

    try:
        json_data = cast(dict[str, Any], json.loads(str(llm_response.answer)))
    except json.JSONDecodeError as exc:
        logger.exception(
            "Invalid JSON returned by LLM",
            llm_output=llm_response.answer,
        )
        raise exc from exc

    logger.debug("Data from LLM", response=llm_response)

    return LLMExtractedImageData(
        schema=schema_generator.schema(
            data=(
                schema_generator.decode_json(data=json_data)
                if schema_generator.encoded
                else json_data
            )
        ),
        llm_response=llm_response,
    )
