from __future__ import annotations
from typing import ClassVar, TypedDict, NotRequired, Any
from flask import Response
from abc import ABC, abstractmethod


class Frame(TypedDict):
    path: str
    choices: NotRequired[list[str]]


class SingleTest(TypedDict):
    start_pause_ms: NotRequired[float]  # show gray screen before
    check_time_ms: NotRequired[float]  # hide choices in
    frames: list[Frame]
    choices: NotRequired[list[str]]  # global choices
    gap: NotRequired[str]  # gap size, for example "100px on 8em"
    text: NotRequired[str]  # text that is shown above question


class Scenario(ABC):
    # optional documentation for the test is HTML format
    __doc__: ClassVar[str] | None

    # Optional CSS rules
    CSS: ClassVar[str] | None

    def __init__(self): ...

    # raise IndexError when test is over
    @abstractmethod
    def item_for_user(self, seed: int, idx: int) -> SingleTest: ...

    # for loading images
    @abstractmethod
    def file(self, path: str) -> Response: ...

    # list test items for debugging
    @abstractmethod
    def items(self) -> Response: ...

    # in place method for removing data before writing it to `answers.ldj`
    @staticmethod
    def filter(answer: dict[str, Any]):
        # remote test.text
        if "test" in answer:
            answer["test"].pop("text", None)
