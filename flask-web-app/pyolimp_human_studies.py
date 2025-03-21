"""

Request scenario:

* User enters username
* User requests first question /study/<scenario_name>/<case_name>/0
* Server answers with json of test properties
*

"""

from __future__ import annotations
from datetime import datetime
import json
from flask import Flask
from flask import request, render_template, Response, abort
from scenarios import Scenario


app = Flask(__name__)
from pathlib import Path

import importlib

scenarios = [
    path.stem
    for path in (Path(__file__).parent / "scenarios").glob("*.py")
    if path.name != "__init__.py"
]


case_instances: dict[tuple[str, str], Scenario] = {}


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
        and issubclass(value, Scenario)
        and value is not Scenario
    ]
    links = [(f"{case}/", case.capitalize()) for case in cases]
    return render_template(
        "links_list.html",
        title=f"Cases for {scenario.capitalize()}",
        links=links,
    )


def get_case_instance(scenario_name: str, case_name: str) -> Scenario:
    key = scenario_name, case_name
    if key in case_instances:
        return case_instances[key]
    if scenario_name not in scenarios:
        # print(f"no scenario {scenario_name}")
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario_name}")
    case_cls = getattr(module, case_name, None)
    if case_cls is None or not issubclass(case_cls, Scenario):
        # print(f"no case scenarios.{scenario_name}.{case_name}")
        abort(404)
    isinstance = case_instances[key] = case_cls()
    return isinstance


@app.route("/study/<scenario_name>/<case_name>/")
def study_scenario_case(scenario_name: str, case_name: str) -> str:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return render_template(
        "index.html", doc=case.__doc__, css=getattr(case, "CSS", None)
    )


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
    seed = int(data["seed"])
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    if data.get("answer"):
        now = datetime.now()
        answer = data["answer"]
        answer["scenario_name"] = scenario_name
        answer["case_name"] = case_name
        answer["username"] = username
        answer["seed"] = seed
        answer["ts"] = f"{now:%Y%m%dT%H%M%S}"
        filter = getattr(case, "filter")
        filter(answer)
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
        return case.item_for_user(seed=seed, idx=index)
    except IndexError:
        return {"finished": True, "text": "Thanks for participating!"}


@app.route("/study/<scenario_name>/<case_name>/file/<path:file_path>")
def file(scenario_name: str, case_name: str, file_path: str) -> Response:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return case.file(file_path)
