import collections
import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Sequence, override

import redis

from llm_toolkit.agentic.runtime.event import AgentRuntimeEvent


class AgentRuntimeTaskBroker(ABC):
    @abstractmethod
    def get_display_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def notify(self, *, event: AgentRuntimeEvent, queue: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_event(self, *, queues: Sequence[str]) -> AgentRuntimeEvent:
        raise NotImplementedError


class InMemoryAgentRuntimeTaskBroker(AgentRuntimeTaskBroker):
    def __init__(self) -> None:
        self.queue_registry = collections.defaultdict[
            str, Queue[AgentRuntimeEvent]
        ](lambda: Queue[AgentRuntimeEvent]())
        self.cond = threading.Condition()

    @override
    def get_display_name(self) -> str:
        return ":memory:"

    @override
    def notify(self, *, event: AgentRuntimeEvent, queue: str) -> None:
        with self.cond:
            self.queue_registry[queue].put(event)
            self.cond.notify()

    @override
    def get_event(self, *, queues: Sequence[str]) -> AgentRuntimeEvent:
        while True:
            with self.cond:
                for queue_name in queues:
                    if not self.queue_registry[queue_name].empty():
                        return self.queue_registry[queue_name].get(block=False)
                self.cond.wait()


class RedisAgentRuntimeTaskBroker(AgentRuntimeTaskBroker):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )

    @override
    def get_display_name(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"

    @override
    def notify(self, *, event: AgentRuntimeEvent, queue: str) -> None:
        self.redis.lpush(queue, event.model_dump_json())

    @override
    def get_event(self, *, queues: Sequence[str]) -> AgentRuntimeEvent:
        while True:
            event = self.redis.brpop(
                keys=list(queues),
                timeout=10,
            )
            if event:
                return AgentRuntimeEvent.model_validate_json(
                    event[1],  # type: ignore
                )

    def __getstate__(self) -> object:
        return (self.host, self.port, self.db)

    def __setstate__(self, state: object) -> None:
        host, port, db = state  # type: ignore
        self.host = host  # type: ignore
        self.port = port  # type: ignore
        self.db = db  # type: ignore
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )
