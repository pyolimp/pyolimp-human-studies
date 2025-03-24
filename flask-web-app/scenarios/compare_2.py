from __future__ import annotations
from typing import Generator, Iterator
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory, jsonify
from . import Scenario, SingleTest


class Test1(Scenario):
    """
    Информация о тесте.
    """

    CSS = """img {width: 37vw;}"""

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
            "check_time_ms": 5000,
            "frames": [
                {"path": path, "choices": [f"choice {idx}"]}
                for idx, path in enumerate(paths)
            ],
            "choices": [
                "left",
                "right",
                "unknown",
            ],
        }
        return ret

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


WELCOME = """
<script>
document.addEventListener("DOMContentLoaded", () => {
    const login_btn= document.getElementById("login_btn");
    login_btn.addEventListener("click", () => {
        login_btn.requestFullscreen();
    });
});
</script>
<style>
.container {
    background-color: #aaa;
    color: black;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width:50%;
    min-width:300px;
}
h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
    font-size: 28px;
}
h2 {
    color: #000040;
    margin-top: 25px;
    font-size: 22px;
}
ul {
    padding-left: 20px;
}
li {
    margin-bottom: 8px;
}
.important-note {
    /*background-color: #f8f4e5;*/
    border-left: 4px solid #f39c12;
    padding: 15px;
    margin: 20px 0;
}
.options {
    /*background-color: #e8f4f8;*/
    padding: 15px;
    border-radius: 5px;
}
.options ul {
    list-style-type: none;
    padding-left: 10px;
}
.options li:before {
    content: "• ";
    color: #000040;
}
</style>

<div class="container">
    <h1>Эксперимент по изучению восприятия изображений</h1>

    <p>Добро пожаловать в эксперимент по обработке изображений для лучшего восприятия слабовидящими!</p>
    <p>
      На экране вам будет показано исходное изображение (в середине)
      и два варианта, как его может увидеть слабовидящий человек. Вам необходимо
      выбрать, какой из вариантов (справа или слева) вы предпочли бы видеть
      вместо исходного (в середине), если бы были слабовидящим.
    </p>

    <p>
      На каждый выбор у вас будет 5 секунд. После того, как они закончатся,
      вам будут предложены варианты ответа: предпочитаете ли вы левое,
      правое или не можете решить, какое предпочесть.
      Всего вам предстоит пройти серию из 30 сравнений.
    </p>

    <div class="important-note">
        <h2>Важно:</h2>
        <ul>
            <li>Эксперимент полностью анонимен. Вы можете указать псевдоним, если хотите, но это не обязательно.</li>
            <li>Проходить эксперимент следует только на компьютере или ноутбуке —
                использование телефонов или планшетов запрещено, чтобы избежать искажений восприятия.</li>
            <li>Если вы носите очки, проходить эксперимент надо в них.</li>
        </ul>
    </div>
</div>
"""

QUESTION = """<p style="width:fit-content;margin:0 auto 5em;">
Пожалуйста, внимательно сравните оба изображения.<br>
Оцените их визуальное качество — чёткость, естественность, наличие или отсутствие артефактов.<br>
Выберите то изображение, которое выглядит для вас более качественным и комфортным для восприятия.<br>
Если разница неочевидна или вы не можете выбрать, нажмите кнопку «Затрудняюсь ответить».</p>"""

CSS_2_PICTURES = """img {width: 37vw;}"""
CSS_3_PICTURES = """img {width: 26vw;}"""


class TestCompareRVIMethods(Scenario):
    __doc__ = WELCOME
    CSS = CSS_3_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/rvi_compare_methods"
        )
        all_test_paths = list(self._root.rglob("**/pair_*/"))
        all_test_paths.sort()
        self._items = list(
            paths
            for test_path in all_test_paths
            for paths in combinations(
                [
                    p.relative_to(self._root)
                    for p in test_path.glob("*.png")
                    if p.name != "target.png"
                ],
                2,
            )
        )

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        random = Random(seed)
        paths = list(random.sample(self._items, k=30)[idx])
        random.shuffle(paths)

        ret: SingleTest = {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {
                    "path": str(paths[0].with_name("target.png")),
                },
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": [
                "Не знаю",
            ],
            "text": QUESTION,
        }
        return ret

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


class TestCompareRVIMetrict(Scenario):
    __doc__ = WELCOME
    CSS = CSS_2_PICTURES

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
        random = Random(seed)
        paths = list(random.sample(self._items, k=30)[idx])
        random.shuffle(paths)
        ret: SingleTest = {
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

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


class TestCompareRVICorr(Scenario):
    __doc__ = WELCOME
    CSS = CSS_2_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/dataset_compare_rvi_corr"
        )
        all_test_paths = [
            path for path in self._root.iterdir() if path.is_dir()
        ]
        all_test_paths.sort()

        def create_checks(
            paths: Generator[Path, None, None],
        ) -> Iterator[tuple[Path, Path]]:
            for path in paths:
                if path.name == "ms-ssim.png":
                    continue
                path = path.relative_to(self._root)
                yield (path, path.with_name("ms-ssim.png"))

        self._items = list(
            item
            for test_path in all_test_paths
            for item in create_checks(test_path.glob("*.png"))
        )

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        random = Random(seed)
        paths = list(random.sample(self._items, k=30)[idx])
        random.shuffle(paths)
        ret: SingleTest = {
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

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )
