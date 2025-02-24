"""

Request scenario:

* User enters username
* User requests first question /study/<scenario_name>/<case_name>/0
* Server answers with json of test properties
*

"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from flask import Flask
from flask import request, render_template, Response, abort
import json


app = Flask(__name__)
from pathlib import Path

import importlib

scenarios = [
    path.stem for path in (Path(__file__).parent / "scenarios").glob("*.py")
]


@runtime_checkable
class TestCase(Protocol):
    def __init__(self): ...

    # raise IndexError when test is over
    def item_for_user(self, seed: int, idx: int) -> Any: ...

    def file(self, path: str) -> Response: ...


case_instances: dict[tuple[str, str], TestCase] = {}


@app.route("/")
def list_scenarios() -> str:
    links = [
        (f"study/{scenario}/", scenario.capitalize()) for scenario in scenarios
    ]
    return render_template("links_list.html", title="Scenarios", links=links)


@app.route("/study/<scenario>/")
def list_cases(scenario: str) -> str:
    if scenario not in scenarios:
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario}")
    cases = [
        key
        for key, value in module.__dict__.items()
        if not key.startswith("__")
        and isinstance(value, type)
        and issubclass(value, TestCase)
    ]
    links = [(f"{case}/", case.capitalize()) for case in cases]
    return render_template(
        "links_list.html",
        title=f"Cases for {scenario.capitalize()}",
        links=links,
    )


def get_case_instance(scenario_name: str, case_name: str) -> TestCase:
    key = scenario_name, case_name
    if key in case_instances:
        return case_instances[key]
    if scenario_name not in scenarios:
        # print(f"no scenario {scenario_name}")
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario_name}")
    case_cls = getattr(module, case_name, None)
    if case_cls is None or not issubclass(case_cls, TestCase):
        # print(f"no case scenarios.{scenario_name}.{case_name}")
        abort(404)
    isinstance = case_instances[key] = case_cls()
    return isinstance


@app.route("/study/<scenario_name>/<case_name>/")
def study_scenario_case(scenario_name: str, case_name: str) -> str:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return render_template("index.html", doc=case.__doc__)


@app.route("/study/<scenario_name>/<case_name>/items")
def study_scenario_case_items(scenario_name: str, case_name: str) -> Response:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    if not hasattr(case, "items"):
        abort(404)
    return case.items()


@app.post("/study/<scenario_name>/<case_name>/<int:index>")
def study_scenario_case_index(
    scenario_name: str, case_name: str, index: int
) -> Response:
    data = request.json
    username = str(data["username"])
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    if data.get("answer"):
        answer = data["answer"]
        answer["scenario_name"] = scenario_name
        answer["case_name"] = case_name
        answer["username"] = username
        filter = getattr(case, "filter")
        if filter is None:
            filter = lambda d: d.pop("text", None)
        answer = filter(answer)
        with open("answers.ldj", "a") as ldj:
            ldj.write(
                json.dumps(
                    data["answer"],
                    separators=(",", ":"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
    try:
        seed = int(data["seed"])
        return case.item_for_user(seed=seed, idx=index)
    except IndexError:
        return {"finished": True, "text": "Thanks for participating!"}


@app.route("/study/<scenario_name>/<case_name>/file/<path:file_path>")
def file(scenario_name: str, case_name: str, file_path: str) -> Response:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return case.file(file_path)
