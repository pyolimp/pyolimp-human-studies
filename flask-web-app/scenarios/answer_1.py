from __future__ import annotations
import json
from typing import Generator, Iterator
from pathlib import Path
from itertools import combinations
from random import Random
from flask import Response, send_from_directory, jsonify
from . import Scenario, SingleTest


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



WELCOME = (
    WELCOME_CSS
    + """

    
    <div class="container">

    <h1>Эксперимент по решению задач Бонгарда основанных на ДЗЗ</h1>

    <p>Добро пожаловать в эксперимент по решению задач Бонгарда основанных на дистанционном зондировании Земли!</p>
    <p>
      На экране вам будут показываться задачи Бонгарда: Каждая задача содержит 2 набора изображений: левый и правый.
      При этом каждый набор имеет общее свойство, которое объединяет изображения одного набора, но отличает от другого.
      Вы должны будете определить это отличие для каждой задачи и напечатать ответ в соотвествующей графе
    </p>

    <p>
      Вам предстоит решить <b>20 задач</b> 
    </p>

    <div class="important-note">
        <h2>Важно:</h2>
        <ul>
            <li>Эксперимент полностью анонимен.</li>
            <li>Проходить эксперимент следует только на компьютере или ноутбуке —
                использование телефонов или планшетов запрещено, чтобы избежать искажений восприятия.</li>
        </ul>
    </div>
</div>
"""
)
EXAMPLE = """
Пример (задача Бонгарда №107):
Правильный ответ: на рисунке выше у фигур слева ровно 3 «простых» (прямых) стороны, а у фигур справа — ровно 3 «сложных»
"""

INTROTWO = """

Инструкция
Вам предстоит решить 20 задач. В каждой из задач будет две группы изображений, по 6 изображений в каждой: слева и справа. Изображения представляют собой снимки дистанционного зондирования Земли: снимки со спутника или летательных аппаратов. В каждой группе изображения объединены общим признаком, и этим же признаком две группы отличаются между собой.

В ответе вам нужно в свободной форме как можно точнее сформулировать, в чем отличие двух групп картинок.

По истечении 60 секунд появится возможность кнопка «Не знаю» — вы можете нажать ее, чтобы перейти к следующей задаче. Задачи появляются последовательно, вернуться к ранее решенной задаче нельзя.

"""

class Bongard(Scenario):
    """
   
    """

    CSS = """
    img {width: 50vw;}
    [name="donotknow"] {
        visibility: hidden;
    }
    .visible {
        visibility: initial;
    }

    """
    __doc__ = WELCOME



    def __init__(self) -> None:
        self._root = Path("/mnt/storage/bong_bench/pyolimp-human-studies/data/problems")
        self._json_dir = Path("/mnt/storage/bong_bench/pyolimp-human-studies/data/batches")
        self._example_image = ("demonstration_test_1.png")
        self._items = {}

    def file(self, path: str) -> Response:
        return send_from_directory(self._root, path)
    
    def _read_next_batch(self):
        for _ in range(10):
            filename = next(self._json_dir.glob("*.json"))
            try:
                new_path = filename.rename(self._json_dir / (filename.stem + '.donejson'))
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
                {
                    "type": "html",
                    "html": INTROTWO
                },
                {
                    "type": "text",
                    "label": "Возраст",
                    "name": "solution",
                    "required": True
                },
                {
                    "type": "choice",
                    "label": "Пол",
                    "name": "sex",
                    "choices": [
                        {"label": "Мужской", "value": "m"},
                        {"label": "Женский", "value": "f"},
                    ]
                },
                {
                    "type": "text",
                    "label": "Род деятельности",
                    "name": "occupation",
                    "required": True
                },
                {
                    "type": "submit",
                    "value": "Продолжить",
                    "name": "continue",
                },
            ],
        }
    def example_1(self) -> SingleTest:
        example_path = self._example_image
        return {
            "frames": [           
                {"path": example_path}
            ],
            "inputs": [
                {
                    "type": "html",
                    "html": EXAMPLE
                },
                {
                    "type": "button",
                    "value": "Перейти к решению задач",
                    "name": "continue"
                }
            ]
        }
    
    @staticmethod
    def _thankyou() -> SingleTest:
        return {
            "frames": [],
            "inputs": [

                {
                    "type": "html",
                    "html": "example text"
                },
                {
                    "type": "button",
                    "value": "Конец",
                    "name": "continue",
                },
                {
                    "type": "text",
                    "label": "Ваш email",
                    "name": "email",
                }
            ]
        }
    def item_for_user(self, seed: int, idx: int) -> SingleTest:
        
        if idx == 0:
            try:
                self._items[seed] = self._read_next_batch()
            except Exception as e:
                return {
                    "frames": [
                        {"path": ""}
                    ],
                    "inputs": [
                        {
                            "type": "text",
                            "label": f"Тесты закончились - '{e}'",
                            "name": "text",
                        },
                    ]
                    }
            else:
                return self._userinfo()
            
        if idx == 1:
            return self.example_1()  
        if idx == len(self._items[seed]["collages"]) + 1:
            return self._thankyou()

        path = self._items[seed]["collages"][idx - 1]
        ret: SingleTest = {
            "frames": [
                {"path": path}
            ],
            "inputs": [
                {
                    "type": "text",
                    "label": "В чем разница между наборами?",
                    "name": "solution",
                    "required": True,
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
            ]
        }
        return ret
    

    def items(self) -> Response:
        return jsonify(list(map(str, self._items)))

