from __future__ import annotations
from typing import ClassVar, TypedDict, Any, Literal
from flask import Response
from abc import ABC, abstractmethod
from typing_extensions import NotRequired


class Frame(TypedDict):
    path: str
    choices: NotRequired[list[str]]


class Input(TypedDict):
    type: Literal["text"]
    required: NotRequired[Literal[True]]
    label: NotRequired[str]
    name: str


class Submit(TypedDict):
    type: Literal["submit"]
    label: NotRequired[str]
    name: str
    value: str


class Button(TypedDict):
    type: Literal["button"]
    label: NotRequired[str]
    name: str
    value: str


class HTML(TypedDict):
    type: Literal["html"]
    label: NotRequired[str]
    html: str


class ChoiceItem(TypedDict):
    label: NotRequired[str]
    value: str


class Choice(TypedDict):
    type: NotRequired[Literal["choice"]]
    name: str
    label: NotRequired[str]
    choices: list[ChoiceItem]


class SingleTest(TypedDict):
    start_pause_ms: NotRequired[float]  # show gray screen before
    check_time_ms: NotRequired[float]  # hide choices in
    frames: list[Frame]
    choices: NotRequired[list[str]]  # global choices
    inputs: NotRequired[
        list[Input | Submit | HTML | Choice | Button]
    ]  # new choices
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
