from __future__ import annotations
from typing import TypedDict, NotRequired
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory


class Frame(TypedDict):
    path: str
    choices: list[str]


class SingleTest(TypedDict):
    start_pause_ms: NotRequired[float]
    frames: list[Frame]
    choices: list[str]


class Test1:
    """
    Информация о тесте.
    """

    def __init__(self) -> None:
        self._root = Path(__file__).parent / "test_images"
        all_paths = self._root.glob("*.*")
        all_image_paths = [
            path.name  # make it json serializable
            for path in all_paths
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        all_image_paths.sort()
        self._items = list(combinations(all_image_paths, 2))

    def file(self, path: str) -> Response:
        print("serving", path, "from", self._root)
        return send_from_directory(self._root, path)

    def item_for_user(self, username: str, idx: int) -> SingleTest:
        paths = Random(username).sample(self._items, k=5)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [{"path": path} for path in paths],
            "choices": [
                "left",
                "right",
                "all good",
                "all bad",
                "unknown",
            ],
        }
        return ret