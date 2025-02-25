from __future__ import annotations
from typing import Iterable, TypedDict
from json import loads
from collections import defaultdict
from scenarios.compare_2 import SingleTest


class Answer(TypedDict):
    case_name: str
    test: SingleTest
    response: str


def iter_answers(json_path: str) -> Iterable[Answer]:
    with open(json_path) as f:
        for line in f:
            yield loads(line)


class Case:
    def __init__(self) -> None:
        self._results: dict[tuple[str, ...], list[float]] = {}

    def add_answer(self, answer: Answer):
        test = answer["test"]
        response = answer["response"]
        test["frames"].sort(key=lambda frame: frame["path"])
        key = tuple([frame["path"] for frame in test["frames"]])
        if key not in self._results:
            self._results[key] = [0] * len(key)
        if response == "Не знаю":
            for answer_idx in range(len(key)):
                self._results[key][answer_idx] += 1 / len(key)
        else:
            answer_idx = next(
                idx
                for idx, frame in enumerate(test["frames"])
                if response in frame["choices"]
            )
            self._results[key][answer_idx] += 1

    def rows(self):
        for result in self._results.items():
            yield result

    def total(self):
        total = 0.0
        for result in self._results.values():
            total += sum(result)
        return total


def process(json_path: str):
    cases = defaultdict[str, Case](Case)
    for answer in iter_answers(json_path):
        cases[answer["case_name"]].add_answer(answer)
    for case_name, case in cases.items():
        print(f"{case_name}")
        for row in case.rows():
            print(row)
        print("total", case.total())


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    process(args.json_path)


if __name__ == "__main__":
    main()
