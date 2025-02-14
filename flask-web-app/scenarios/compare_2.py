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


WELCOME = """<p>
Добро пожаловать в эксперимент по изучению восприятия изображений!<br>
В ходе этого исследования вам предстоит пройти серию из 10 вопросов,<br>
каждый из которых представляет собой сравнение двух изображений. <br>
Ваш выбор поможет нам лучше понять, как люди воспринимают визуальную информацию.
</p>

<p>Как проходит эксперимент:</p>

<ul>
<li>На каждом вопросе вам будут показаны два изображения.
<li>У вас есть три варианта ответа:
<li>Левое изображение – если вы предпочитаете левое изображение.
<li>Правое изображение – если вы предпочитаете правое изображение.
<li>Не знаю – если вам сложно определиться.
<li>Ваши ответы будут использоваться исключительно в исследовательских целях<br>
    и останутся <b>анонимными</b>.
</ul>"""

QUESTION = (
    """<p style="width:fit-content;margin:0 auto 5em;">
Пожалуйста, внимательно сравните оба изображения.<br>
Если вы предпочитаете одно из них, нажмите соответствующую кнопку.<br>
Если выбор затруднителен, воспользуйтесь кнопкой «Не знаю».<p>""",
)


class TestCompareRVIMethods:
    __doc__ = WELCOME

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/dataset_compare_rvi_methods"
        )
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
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        paths = Random(seed).sample(self._items, k=10)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": [
                "Не знаю",
            ],
            "text": QUESTION,
        }
        return ret


class TestCompareRVIMetrict:
    __doc__ = WELCOME

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/dataset_compare_rvi_metrics"
        )
        self._items = list(
            tuple(
                [path.relative_to(self._root) for path in path.glob("*.png")]
            )
            for path in self._root.iterdir()
            if path.is_dir()
        )
        for item in self._items:
            assert len(item) == 2, item

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        paths = Random(seed).sample(self._items, k=10)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": [
                "Не знаю",
            ],
            "text": QUESTION,
        }
        return ret


class TestCompareRVICorr:
    __doc__ = WELCOME

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/dataset_compare_rvi_corr"
        )
        all_test_paths = [
            path for path in self._root.iterdir() if path.is_dir()
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
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        paths = Random(seed).sample(self._items, k=10)[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": [
                "Не знаю",
            ],
            "text": QUESTION,
        }
        return ret
