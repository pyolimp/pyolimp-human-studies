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


FULL_SCREEN = """<script>
document.addEventListener("DOMContentLoaded", () => {
    const login_btn= document.getElementById("login_btn");
    login_btn.addEventListener("click", () => {
        login_btn.requestFullscreen();
    });
});
</script>"""


WELCOME_CSS = """<style>
.container {
    background-color: #aaa;
    color: black;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 50%;
    min-width: 300px;
    margin: auto;
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
    border-left: 4px solid #f39c12;
    padding: 15px;
    margin: 20px 0;
}
.options {
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
</style>"""

WELCOME = (
    WELCOME_CSS
    + """
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
      На каждый выбор у вас будет 7 секунд. После того, как они закончатся,
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
)

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
            "check_time_ms": 7000,
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
        }
        return ret

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


class TestCompareRVIMetrics(Scenario):
    __doc__ = WELCOME
    CSS = CSS_3_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/rvi_compare_metrics"
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
                    if p.name != "blurred.png"
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
            "check_time_ms": 7000,
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
        }
        return ret

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


class TestCompareCorrMSSim(Scenario):
    __doc__ = WELCOME
    CSS = CSS_3_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/human_studies_corr_mssim"
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
                    if p.name != "blurred.png"
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
            "check_time_ms": 7000,
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
        }
        return ret

    def items(self) -> Response:
        return jsonify(
            [[str(path) for path in paths] for paths in self._items]
        )


class TestRankCorrMSSimm(Scenario):
    __doc__ = "Вам скажут, что надо делать"
    CSS = CSS_3_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/human_studies_corr_mssim"
        )
        corrs = sorted(self._root.rglob("**/pair_*/corr_cf1.2000_white.png"))[
            :15
        ]
        self._items: list[Path] = []
        for corr in corrs:
            corr = corr.relative_to(self._root)
            self._items.append(corr)
            self._items.append(corr.with_name("ms-ssim.png"))

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        path = self._items[idx]
        ret: SingleTest = {
            "start_pause_ms": 1500,
            "check_time_ms": 7000,
            "frames": [
                {"path": str(path.with_name("target.png"))},
                {"path": str(path), "choices": ["1", "2", "3", "4", "5"]},
            ],
        }
        return ret

    def items(self) -> Response:
        return jsonify(list(map(str, self._items)))


class TestCVD(Scenario):
    __doc__ = (
        WELCOME_CSS
        + """<div class="container">
    <h1>Эксперимент по оценке алгоритмов улучшения восприятия при нарушениях цветового зрения</h1>

    <p>Добро пожаловать в эксперимент по оценке алгоритмов улучшения цветового восприятия для людей с нарушением различения красного и зелёного цветов (дальтинизмом)!</p>
    <p>
      Для людей с дальтонизмом зачастую может быть проблемой не только различить, где красный цвет, а где зеленый, но и заметить границу
      между разными цветами. На экране вам будет показано исходное изображение (в центре) и два варианта (слева и справа),
      симулирующих видение человека с нарушениями цветового зрения, обработанных разными алгоритмами компенсации.
    </p>

    <p>
      Ваша задача:
      <ul>
        <li>Рассмотрите внимательно исходное изображение. Найдите границы между зеленым и красным цветами. Посмотрите, насколько хорошо
        объекты этих цветов отличимы друг от друга на двух вариантах симуляции — слева и справа.

        <li> Рассмотрите внимательно симулированные изображения. Не возникло ли на них ложных различий цветов, которых не было на исходном
        изображении?</li>

        <li>На основе всего увиденного выберите, какой вариант предобработки — правый или левый — вы бы предпочли на месте человека с
        нарушениями цветового зрения?</li>
      </ul>
    </p>

    <div class="important-note">
        <h2>Важно:</h2>
        <ul>
            <li>Эксперимент полностью анонимен. Вы можете указать псевдоним, если хотите, но это не обязательно.</li>
            <li>Проходить эксперимент следует только на компьютере или ноутбуке — использование телефонов и планшетов запрещено.</li>
            <li>Если вы носите очки, проходите эксперимент в них.</li>
        </ul>
    </div>
</div>
"""
    )
    TEXT = """<p style="width:fit-content;margin:0 auto 5em;">
Пожалуйста, внимательно сравните изображения справа и слева.<br>
Оцените, какой вариант обработки лучше передает различия цветов там, где на оригинальном изображении границы красного и зеленого цветов,<br>
и не вносит лишних искажений там, где таких различий на исходном изображении нет.<br>
Выберите тот вариант, который обеспечивает более качественную компенсацию.<br>
Если разница неочевидна или вы не можете выбрать, нажмите кнопку «Затрудняюсь ответить».
</p>
"""
    CSS = CSS_3_PICTURES

    def __init__(self) -> None:
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/human_studies_cvd"
        )
        all_test_paths = list(self._root.rglob("**/pair_*/"))
        all_test_paths.sort()
        self._items = list(
            paths
            for test_path in all_test_paths
            for paths in combinations(
                [
                    p.relative_to(self._root)
                    for p in test_path.glob("*_sim.png")
                ],
                2,
            )
        )

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        random = Random(seed)
        paths = list(random.sample(self._items, k=15)[idx])
        random.shuffle(paths)

        ret: SingleTest = {
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
            "text": self.TEXT,
        }
        return ret

    def items(self) -> Response:
        return jsonify(list(map(str, self._items)))
