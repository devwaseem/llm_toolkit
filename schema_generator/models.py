import copy
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar, cast, override


class Field[T: None | int | float | str | bool | datetime | list[Any]](ABC):
    def __init__(
        self,
        description: str,
        *,
        value_type: str,
        extra_kwargs: dict[str, Any] | None = None,
        default: T | None = None,
    ) -> None:
        self._value: T | None = None
        self.description = description
        self.value_type = value_type
        self.format = format
        self.default: T | None = default
        self.extra_args = extra_kwargs

    def set_value(self, value: T | None) -> None:
        self._value = value

    @abstractmethod
    def get_value(self) -> T | None:
        raise NotImplementedError


class IntegerField(Field[int]):
    def __init__(
        self,
        description: str = "",
        *,
        extra_kwargs: dict[str, Any] | None = None,
        default: int | None = None,
    ) -> None:
        super().__init__(
            description=description,
            value_type="integer",
            extra_kwargs=extra_kwargs,
            default=default,
        )

    @override
    def get_value(self) -> int | None:
        return int(self._value) if self._value is not None else self.default


class FloatField(Field[float]):
    def __init__(
        self,
        description: str = "",
        *,
        extra_kwargs: dict[str, Any] | None = None,
        default: float | None = None,
    ) -> None:
        super().__init__(
            description=description,
            value_type="float",
            extra_kwargs=extra_kwargs,
            default=default,
        )

    @override
    def get_value(self) -> float | None:
        return float(self._value) if self._value is not None else self.default


class StringField(Field[str]):
    def __init__(
        self,
        description: str = "",
        *,
        extra_kwargs: dict[str, Any] | None = None,
        default: str | None = None,
    ) -> None:
        super().__init__(
            description=description,
            value_type="string",
            extra_kwargs=extra_kwargs,
            default=default,
        )

    @override
    def get_value(self) -> str | None:
        return str(self._value) if self._value is not None else self.default


class DateTimeField(Field[str]):
    def __init__(
        self,
        description: str = "",
        *,
        extra_kwargs: dict[str, Any] | None = None,
        default: str | None = None,
    ) -> None:
        if extra_kwargs is None:
            extra_kwargs = {}

        extra_kwargs["format"] = "isoformat"
        extra_kwargs["example"] = "2000-10-31T01:30:00.000Z"
        super().__init__(
            description=description,
            value_type="datetime with timezone",
            extra_kwargs=extra_kwargs,
            default=default,
        )

    @override
    def get_value(self) -> str | None:
        return self._value if self._value is not None else self.default

    def get_value_as_datetime(self) -> datetime | None:
        if datetime_str := self.get_value():
            return datetime.fromisoformat(datetime_str)
        return None


class BooleanField(Field[bool]):
    def __init__(
        self,
        description: str = "",
        *,
        default: bool | None = None,
    ) -> None:
        super().__init__(
            description=description,
            value_type="boolean",
            default=default,
        )

    def get_value(self) -> bool | None:
        return self._value


class ListField[F: LLMSchemaModel | Field[Any]]:
    cls: type[F]
    kwargs: dict[str, Any] | None = None
    _value: list[F]

    def __init__(
        self,
        cls: type[F],
        *,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._value = []
        self.cls = cls
        self.kwargs = kwargs

    def set_value(self, value: list[F]) -> None:
        self._value = value

    def get_value(self) -> list[F]:
        return self._value


class LLMSchemaModel:
    def __init__(
        self,
        description: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        self.description = description
        if data:
            self.load_data(data=data)

    def load_data(self, *, data: dict[str, Any]) -> None:
        for attr in dir(self):
            if attr.startswith("_"):
                continue

            if attr not in data:
                continue

            value = data[attr]

            if not value:
                continue

            field = copy.deepcopy(getattr(self, attr))
            if isinstance(field, Field):
                field.set_value(value)
            elif isinstance(field, ListField):
                instances = [
                    self._create_instance(
                        cls=field.cls,
                        kwargs=field.kwargs or {},
                        data=item,
                    )
                    for item in value
                ]
                field.set_value(instances)

            elif isinstance(field, LLMSchemaModel):
                field.load_data(data=value)

            setattr(self, attr, field)

    def _create_instance(
        self,
        *,
        cls: type["LLMSchemaModel"] | type[Field[Any]],
        kwargs: dict[str, Any],
        data: Any,
    ) -> "LLMSchemaModel | Field[Any]":
        instance = cls(**kwargs)
        if isinstance(instance, LLMSchemaModel):
            instance.load_data(data=data)
            return instance

        if isinstance(instance, Field):
            instance.set_value(data)
            return instance

        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        value = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue

            attr_value = getattr(self, attr)

            if isinstance(attr_value, Field):
                value[attr] = attr_value.get_value()

            elif isinstance(attr_value, LLMSchemaModel):
                value[attr] = attr_value.to_dict()

            elif isinstance(attr_value, ListField):
                item_value_list = []
                for item in attr_value.get_value():
                    if isinstance(item, Field):
                        item_value_list.append(item.get_value())
                    elif isinstance(item, LLMSchemaModel):
                        item_value_list.append(item.to_dict())

                value[attr] = item_value_list

        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_json_safe_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.to_json()))

    def __repr__(self) -> str:
        return self.to_json()


LLMSchemaModelTypeVar = TypeVar("LLMSchemaModelTypeVar", bound=LLMSchemaModel)


class LLMSchemaGenerator(ABC, Generic[LLMSchemaModelTypeVar]):
    def __init__(
        self,
        *,
        schema: type[LLMSchemaModelTypeVar],
        encoded: bool = True,
    ) -> None:
        self.schema = schema
        self.encoded = encoded

    @abstractmethod
    def build_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def decode_json(self, *, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def decoded_schema(self, *, data: dict[str, Any]) -> LLMSchemaModelTypeVar:
        return self.schema(
            data=self.decode_json(data=data),
        )

    @abstractmethod
    def get_example(self) -> str:
        raise NotImplementedError
