from __future__ import annotations
from typing import ClassVar, TypedDict, Any, Literal
from flask import Response
from abc import ABC, abstractmethod
from typing_extensions import NotRequired


class Frame(TypedDict):
    """
    Project `pyolimp-human-studies` is designed to evaluate images.
    This item represents one image with multiple choices
    """

    path: str
    choices: NotRequired[list[str]]


class WithLabel(TypedDict):
    """
    Every added item can have label that is rendered as div
    """

    label: NotRequired[str]


class FormElement(WithLabel):
    name: str
    """
    Name of the form value
    """


class Input(WithLabel):
    """
    Element rendered as a textarea
    """

    type: Literal["text"]
    required: NotRequired[Literal[True]]
    cols: NotRequired[str]
    rows: NotRequired[str]


class Submit(FormElement):
    """
    Button that will submit the form and validate `required` input elements
    """

    type: Literal["submit"]
    value: str


class Button(FormElement):
    """
    Button that will submit the form without validating input elements
    """

    type: Literal["button"]
    value: str


class HTML(WithLabel):
    """
    Insert raw HTML, for example, to show an image or rich text
    """

    type: Literal["html"]
    html: str


class ChoiceItem(TypedDict):
    """
    HTML Option element
    """

    label: str
    "User visible text"
    value: str


class Choice(FormElement):
    """
    HTML Select element with multiple choices
    """

    type: Literal["choice"]
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
