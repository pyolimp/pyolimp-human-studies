from __future__ import annotations
import json
from typing import Generator, Iterator
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory, jsonify
from . import Scenario, SingleTest
from codecs import encode

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
</style>
     <script>
    let timeoutId = undefined;
    function on_create_test_page() {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            document.querySelector('[name="donotknow"]').className = "visible";
        }, 1000 * 60);
    }
</script>
"""


WELCOME = WELCOME_CSS + """

    <div class="container">

    <p>Уважаемый участник,</p>
    <p>Приветствуем вас в веб-интерфейсе эксперимента по зрительному распознаванию образов человеком! Исследование проводится лабораторией №11 ИППИ РАН в исключительно научных целях.</p>
    <h2>Как проходит эксперимент</h2>
    <p>

    Вам предстоит разгадать 20 головоломок на сравнение изображений. Изображения — это снимки дистанционного зондирования Земли (спутниковые или с летательных аппаратов). Решение всех 20 задач займет около 30 минут.

    </p>
    <p>

    Вы можете прекратить участие в эксперименте в любой момент, закрыв вкладку браузера. Если после этого вы захотите вернуться к эксперименту, задачи начнутся с начала.

    </p>

    <h2> Важно</h2>

    <div class="important-note">

        <ul>
            <li>Эксперимент полностью анонимен.</li>
            <li>Пожалуйста, проходите эксперимент только на компьютере или ноутбуке —
                использование телефонов или планшетов запрещено, чтобы избежать искажений восприятия.</li>
        </ul>
    </div>

    <p> Введите любой псевдоним или оставьте случайно сгенерированный. Псевдоним будет использоваться для маркировки ответов.  </p>
</div>
"""
FORMBEGIN = """
    <div class="container">

<li>Чтобы начать эксперимент, заполните, пожалуйста, анкету</li>
<li>Нажимая на кнопку «Начать эксперимент», вы подтверждаете, что соглашаетесь с описанными выше условиями участия в эксперименте.</li>


"""

INTROTWO = """
<div class="container"> <h2>Инструкция</h2>
    <p>

    Вам предстоит решить 20 задач по зрительному распознаванию образов на снимках дистанционного зондирования Земли (аналогии задач Бонгарда).

    </p>

    <p>

    В каждой задаче вы увидите 2 набора изображений, по 6 изображений в каждом: левый и правый. Изображения в левом наборе объединены общим признаком, которым не обладают изображения в правом наборе. И наоборот, изображения в правом наборе обладают признаком, которого нет в изображениях левого набора. Для решения задачи вам нужно сформулировать эти признаки.

    </p>
    <p>

    Если наборы изображений загрузятся частично, обновите, пожалуйста, страницу.

    </p>
    <p>

    По истечении 60 секунд появится кнопка «Не знаю» — вы можете нажать ее, чтобы перейти к следующей задаче. Задачи появляются последовательно, вернуться к предыдущей задаче нельзя.

    </p>
    <p>

    Далее вы увидите примеры задач Бонгарда.

    </p>
"""
EXAMPLEONE = """

<h2>Пример (задача Бонгарда №2):</h2>

<h3>Правильный ответ:
</h3>
<table width=100%><tr><td align=center>

Большие фигуры
</td><td align=center>
Маленькие фигуры
</td></tr></table>

<h3>Также возможен ответ:
</h3>
<table width=100%><tr><td align=center>

Фигуры занимают много места
</td><td align=center>
Фигуры занимают мало места
</td></tr></table>

<h3>Также возможен ответ:
</h3>
<table width=100%><tr><td align=center>

Характерный размер фигуры составляет <br> хотя бы треть стороны изображения
</td><td align=center>
Характерный размер фигуры составляет <br/> меньше трети стороны изображения
</td></tr></table>


"""
EXAMPLETWO = """
<h3>Пример (задача Бонгарда №10):</h3>
<table width=100%><tr><td align=center>
Треугольники
</td><td align=center>
Четырехугольники
</td></tr></table>

<h3>Также возможен ответ:
</h3>
<table width=100%><tr><td align=center>

Отрезки, апроксимирующие стороны фигур, <br> образуют треугольники
</td><td align=center>
Отрезки, апроксимирующие стороны фигур, <br> образуют четырехугольники
</td></tr></table>

"""

END = """
<div class="container"> Спасибо за участие! Если вы хотите узнать ответы, оставьте, пожалуйста, email. Мы отправим их в течение нескольких месяцев, после завершения исследования.

Если вам интересно узнать больше про задачи М. М. Бонгарда, то вы можете прочитать его книгу "Проблема узнавания". В конце книги приведены 100 задач с ответами.
"""


class Bongard(Scenario):
    """ """

    CSS = """
    img {width: 60vw;}
    [name="donotknow"] {
        visibility: hidden;
    }
    .visible {
        visibility: initial;
    }
    #responseform {
        margin-top:1em;
    }
    .submit_ok {
        margin: 1.3em;
    }
    .submit_ok, .button_donotknow {
        text-align: center;
    }
    #start_over {
        display: none;
    }

    """
    __doc__ = WELCOME

    def __init__(self) -> None:
        self._root = Path(
            ...
        )
        self._json_dir = Path(
            ...
        )
        self._example_one = "demonstration_1.png"
        self._example_two = "demonstration_2.png"
        self._items = {}

    def file(self, path: str) -> Response:
        if path not in [self._example_one, self._example_two]:
            path = encode(path, "rot_13")
        return send_from_directory(self._root, path)

    def _read_next_batch(self):
        for _ in range(10):
            filename = next(self._json_dir.glob("*.json"))
            try:
                new_path = filename.rename(
                    self._json_dir / (filename.stem + ".donejson")
                )
            except FileNotFoundError:
                continue
            text = new_path.read_text()
            return json.loads(text)
        raise RuntimeError("programmer error")

    @staticmethod
    def _userinfo() -> SingleTest:
        return {
            "frames": [],
            "inputs": [
                {"type": "html", "html": FORMBEGIN},
                {
                    "type": "text",
                    "label": "Возраст",
                    "name": "age",
                    "required": True,
                    "cols": "5",
                },
                {
                    "type": "choice",
                    "label": "Пол",
                    "name": "sex",
                    "choices": [
                        {"label": "Мужской", "value": "m"},
                        {"label": "Женский", "value": "f"},
                    ],
                },
                {
                    "type": "text",
                    "label": "Род деятельности",
                    "name": "occupation",
                    "cols": "30",
                    "required": True,
                },
                {
                    "type": "submit",
                    "value": "Начать эксперимент",
                    "name": "continue",
                },
            ],
        }

    @staticmethod
    def _instruction() -> SingleTest:
        return {
            "frames": [],
            "inputs": [
                {"type": "html", "html": INTROTWO},
                {
                    "type": "button",
                    "value": "Продолжить",
                    "name": "continue",
                },
            ],
        }

    def example_one(self) -> SingleTest:
        example_path = self._example_one
        return {
            "frames": [{"path": example_path}],
            "inputs": [
                {"type": "html", "html": EXAMPLEONE},
                {
                    "type": "button",
                    "value": "Продолжить",
                    "name": "continue1",
                },
            ],
        }

    def example_two(self) -> SingleTest:
        example_path = self._example_two
        return {
            "frames": [{"path": example_path}],
            "inputs": [
                {"type": "html", "html": EXAMPLETWO},
                {
                    "type": "button",
                    "value": "Перейти к решению задач",
                    "name": "continue",
                },
            ],
        }

    @staticmethod
    def _thankyou() -> SingleTest:
        return {
            "frames": [],
            "inputs": [
                {"type": "html", "html": END},
                {
                    "type": "text",
                    "label": "Ваш email",
                    "name": "email",
                    "cols": "25",
                },
                {
                    "type": "button",
                    "value": "Завершить тестирование",
                    "name": "continue",
                },
            ],
        }

    def item_for_user(self, seed: int, idx: int) -> SingleTest:

        if idx == 0:
            try:
                self._items[seed] = self._read_next_batch()
            except Exception as e:
                return {
                    "frames": [],
                    "inputs": [
                        {
                            "type": "text",
                            "label": f"Тесты закончились - '{e}'",
                            "name": "text",
                        },
                    ],
                }
            else:
                return self._userinfo()
        paths = self._items[seed]["collages"]
        if idx == 1:
            return self._instruction()
        if idx == 2:
            return self.example_one()
        if idx == 3:
            return self.example_two()
        if idx == len(paths) + 4:
            return self._thankyou()
        path = paths[idx - 4]

        ret: SingleTest = {
            "text": f"Задача {idx - 3} из {len(paths)}",
            "frames": [{"path": encode(path, "rot_13")}],
            "inputs": [
                {
                    "type": "html",
                    "html": "<table width=100%><tr><td align=center>",
                },
                {
                    "type": "text",
                    "label": "Признак для левого набора",
                    "name": "left_ans",
                    "required": True,
                    "cols": "44",
                    "rows": "4",
                },
                {
                    "type": "html",
                    "html": "</td><td align=center>",
                },
                {
                    "type": "text",
                    "label": "Признак для правого набора",
                    "name": "right_ans",
                    "required": True,
                    "cols": "44",
                    "rows": "4",
                },
                {
                    "type": "html",
                    "html": "</td></tr></table>",
                },
                {
                    "type": "submit",
                    "value": "Сохранить ответ",
                    "name": "ok",
                },
                {
                    "type": "button",
                    "value": "Не знаю",
                    "name": "donotknow",
                },
            ],
        }
        return ret

    def items(self) -> Response:
        return jsonify(list(map(str, self._items)))
