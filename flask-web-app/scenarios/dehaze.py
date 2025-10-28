from __future__ import annotations
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory, jsonify
from . import Scenario, SingleTest
import json


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
    + """<div class="container">
    <h1>Эксперимент по оценке методов раздымки изображений</h1>

    <p>Добро пожаловать в эксперимент! В нём вам будут показаны изображения,
    обработанные разными алгоритмами удаления дымки.</p>

    <p>
      Для каждой сцены случайно выбирается один из двух синтезов
      (<b>CSNC</b> или <b>CSSO</b>), затем два случайных метода из четырёх:
      hazed, cadcp, dcp, dform.
    </p>

    <p>
      На экране одновременно будут показаны два изображения на 10 секунд.
      Необходимо выбрать изображение которое кажется менее задымленным.
      Если не получается однозначно выбрать менее задымленный вариант - нажмите Не знаю
    </p>

    <div class="important-note">
        <h2>Важно:</h2>
        <ul>
            <li>Эксперимент полностью анонимен.</li>
            <li>Проходить можно только с компьютера или ноутбука.</li>
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


class TestCompareDehazingMethods(Scenario):
    __doc__ = WELCOME

    CSS = """img {width: inherit!;}"""

    def __init__(self):
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/dehaze/_final_dehazing_data"
        )
        self.synths = ["csnc", "csso"]
        self.methods = ["hazed", "cadcp", "dcp", "dform"]

        # формируем все возможные пары заранее
        self._items = []
        for synth in self.synths:
            for method1, method2 in combinations(self.methods, 2):
                files_m1 = list((self._root / synth / method1).glob("*.png"))
                for f1 in files_m1:
                    scene_name = f1.stem.rsplit("_", 1)[0]
                    f2 = (
                        self._root
                        / synth
                        / method2
                        / f"{scene_name}_{method2}.png"
                    )
                    if f2.exists():
                        self._items.append(
                            (
                                f1.relative_to(self._root),
                                f2.relative_to(self._root),
                            )
                        )

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int, n: int = 50) -> SingleTest:
        rng = Random(seed)
        # n случ пар
        n_questions = min(n, len(self._items))
        selected = rng.sample(self._items, k=n_questions)

        # check end
        if idx >= n_questions:
            return jsonify({"finished": True})  # type: ignore

        # now para
        paths = list(selected[idx])
        Random(seed + idx).shuffle(paths)

        return {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": ["Не знаю"],
            "text": "Выбрать изображение которое кажется менее задымленным.<br> Если не получается однозначно выбрать менее задымленный вариант - нажмите Не знаю",
        }

    def items(self) -> Response:
        #
        return jsonify([list(map(str, pair)) for pair in self._items])


class TestCompareDehazingMethods_csso(Scenario):
    __doc__ = WELCOME

    CSS = """img {width: inherit!;}"""

    def __init__(self):
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/dehaze/_final_dehazing_data"
        )
        self.synths = ["csso"]
        self.methods = ["hazed", "cadcp", "dcp", "dform"]

        # формируем все возможные пары заранее
        self._items = []
        for synth in self.synths:
            for method1, method2 in combinations(self.methods, 2):
                files_m1 = list((self._root / synth / method1).glob("*.png"))
                for f1 in files_m1:
                    scene_name = f1.stem.rsplit("_", 1)[0]
                    f2 = (
                        self._root
                        / synth
                        / method2
                        / f"{scene_name}_{method2}.png"
                    )
                    if f2.exists():
                        self._items.append(
                            (
                                f1.relative_to(self._root),
                                f2.relative_to(self._root),
                            )
                        )

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int, n: int = 50) -> SingleTest:
        rng = Random(seed)
        # n случ пар
        n_questions = min(n, len(self._items))
        selected = rng.sample(self._items, k=n_questions)

        # check end
        if idx >= n_questions:
            return jsonify({"finished": True})  # type: ignore

        # now para
        paths = list(selected[idx])
        Random(seed + idx).shuffle(paths)

        return {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": ["Не знаю"],
            "text": "Выбрать изображение которое кажется менее задымленным.<br> Если не получается однозначно выбрать менее задымленный вариант - нажмите Не знаю",
        }

    def items(self) -> Response:
        #
        return jsonify([list(map(str, pair)) for pair in self._items])


class TestCompareDehazingMethods_unic(Scenario):
    __doc__ = WELCOME

    CSS = """img {width: inherit!;}"""
    ANSWERS_FILE = "/home/human_studies/projects/pyolimp-human-studies/flask-web-app/answers.ldj"

    def __init__(self):
        self._root = Path(
            "/home/human_studies/projects/hs_pyolimp_data/dehaze/_final_dehazing_data"
        )
        self.synths = ["csnc", "csso"]
        self.methods = ["hazed", "cadcp", "dcp", "dform"]

        # загружаем все уже существующие пары из answers.ldj
        self.existing_pairs = set()
        try:
            with open(self.ANSWERS_FILE, encoding="utf8") as f:
                for line in f:
                    ans = json.loads(line)
                    frames = ans.get("test", {}).get("frames", [])
                    if len(frames) == 2:
                        paths = [frames[0]["path"], frames[1]["path"]]
                        pair = tuple(sorted(paths))
                        self.existing_pairs.add(pair)
        except FileNotFoundError:
            pass  # если файла ещё нет, просто пусто

        # формируем все возможные пары
        self.all_pairs = []
        for synth in self.synths:
            for method1, method2 in combinations(self.methods, 2):
                files_m1 = list((self._root / synth / method1).glob("*.png"))
                for f1 in files_m1:
                    scene_name = f1.stem.rsplit("_", 1)[0]
                    f2 = (
                        self._root
                        / synth
                        / method2
                        / f"{scene_name}_{method2}.png"
                    )
                    if f2.exists():
                        pair = tuple(
                            sorted(
                                [
                                    str(f1.relative_to(self._root)),
                                    str(f2.relative_to(self._root)),
                                ]
                            )
                        )
                        # добавляем только если пара ещё не встречалась
                        if pair not in self.existing_pairs:
                            self.all_pairs.append(
                                (
                                    str(f1.relative_to(self._root)),
                                    str(f2.relative_to(self._root)),
                                )
                            )

        # Итого all_pairs - пары, которых ещё нет

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)

    def item_for_user(self, seed: int, idx: int, n: int = 50) -> SingleTest:
        rng = Random(seed)
        n_questions = min(n, len(self.all_pairs))
        selected = rng.sample(self.all_pairs, k=n_questions)

        if idx >= n_questions:
            return jsonify({"finished": True})  # type: ignore

        paths = list(selected[idx])
        Random(seed + idx).shuffle(paths)

        return {
            "start_pause_ms": 1500,
            "frames": [
                {"path": str(paths[0]), "choices": ["Левое"]},
                {"path": str(paths[1]), "choices": ["Правое"]},
            ],
            "choices": ["Не знаю"],
            "text": (
                "Выбрать изображение которое кажется менее задымленным.<br>"
                "Если не получается однозначно выбрать менее задымленный вариант - нажмите Не знаю"
            ),
        }

    def items(self) -> Response:
        return jsonify([list(map(str, pair)) for pair in self.all_pairs])
