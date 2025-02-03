from __future__ import annotations
from typing import TypedDict, NotRequired
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory


class Frame(TypedDict):
    path: str
    choices: NotRequired[list[str]]


class SingleTest(TypedDict):
    start_pause_ms: NotRequired[float]
    frames: list[Frame]
    choices: list[str]
    gap: NotRequired[str]  # list "100px on 8em"


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
            if path.is_file()
            and path.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        all_image_paths.sort()
        self._items = list(combinations(all_image_paths, 2))

    def file(self, path: str) -> Response:
        print("serving", path, "from", self._root)
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        paths = Random(seed).sample(self._items, k=5)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [{"path": path} for path in paths],
            "choices": [
                "left",
                "right",
                "unknown",
            ],
        }
        return ret


class Test2:
    """
    Информация о тесте 2.
    """

    def __init__(self) -> None:
        self._root = Path("/home/senyai/Downloads/Telegram Desktop/dataset_1")
        all_test_paths = [
            subpath
            for path in self._root.iterdir()
            if path.is_dir()
            for subpath in path.iterdir()
            if subpath.is_dir()
        ]
        all_test_paths.sort()
        self._items = list(
            paths
            for test_path in all_test_paths
            for paths in combinations(
                [p.relative_to(self._root) for p in test_path.glob("*.png")], 2
            )
        )

    def file(self, path: str) -> Response:
        # print("serving", path, "from", self._root)
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        paths = Random(seed).sample(self._items, k=5)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Left"]},
                {"path": str(paths[1]), "choices": ["Right"]},
            ],
            "choices": [
                "Unknown",
            ],
        }
        return ret
