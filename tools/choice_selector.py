import json
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from llm_toolkit.llm.errors import (
    LLMAPIConnectionError,
    LLMAPITimeoutError,
    LLMInternalServerError,
    LLMJsonResponseDecodingError,
    LLMRateLimitedError,
)
from llm_toolkit.llm.models import (
    LLM,
    LLMInputMessage,
    LLMMessageRole,
    StructuredOutputLLM,
)


class LLMChoice(NamedTuple):
    choice_id: str
    option: str | int | dict[str, Any]


_SYSTEM_MESSAGE = """ 
You are a Intelligent Machine which given a question and
list of choices in JSON, You will pick the right choice and return its ID
in JSON format as explained below. If no option is correct, return null.
Only return the correct id if you have high confidence, return null for low confidence.

Example 1:
Question: What is 10+20
Options:
[{"id": "1", "option": "30"},{"id": "2", "option": "40"},{"id": "3", "option": "100"}]

Output:
{"id": "1"}

Example 2:
Question: What is 20+20
Options:
[{"id": "1", "option": "100"},{"id": "2", "option": "200"},{"id": "3", "option": "300"}]

Output:
{"id": null}
"""  # noqa


class LLMSelectedChoice(BaseModel):
    id: str | None


class _StructuredOutputSupportedLLM(LLM, StructuredOutputLLM, Protocol): ...


@retry(
    retry=retry_if_exception_type(LLMJsonResponseDecodingError),
    reraise=True,
    stop=stop_after_attempt(3),
)
@retry(
    retry=retry_if_exception_type(
        exception_types=(
            LLMAPIConnectionError,
            LLMAPITimeoutError,
            LLMRateLimitedError,
            LLMInternalServerError,
        )
    ),
    reraise=True,
    wait=wait_exponential_jitter(),
    stop=stop_after_delay(300),
)
def llm_pick_choice(
    llm: _StructuredOutputSupportedLLM,
    question: str,
    choices: list[LLMChoice],
) -> LLMChoice | None:
    choices_by_id_dict: dict[str, LLMChoice] = {
        choice.choice_id: choice for choice in choices
    }
    options_json = json.dumps([
        {"id": choice_id, "option": choice.option}
        for choice_id, choice in choices_by_id_dict.items()
    ])
    selected_choice, llm_response = llm.extract(
        messages=[
            LLMInputMessage(
                role=LLMMessageRole.USER,
                content=(f"Question: {question}" f"choices: {options_json}"),
            )
        ],
        schema=LLMSelectedChoice,
    )
    if selected_choice.id is None:
        return None

    return choices_by_id_dict.get(selected_choice.id)
