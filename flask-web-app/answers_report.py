#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from collections import defaultdict
from typing import Iterable, TypedDict
from tabulate import tabulate
import sys
import os
import csv


class Frame(TypedDict, total=False):
    path: str
    choices: list[str]


class SingleTest(TypedDict):
    frames: list[Frame]


class Answer(TypedDict):
    case_name: str
    test: SingleTest
    response: str


def load_answers(json_path: str) -> Iterable[Answer]:
    if json_path == "-":
        f = sys.stdin
    else:
        f = open(json_path, encoding="utf8")
    with f:
        for line in f:
            yield json.loads(line)


def get_method_name(path: str) -> str:
    # Вытащим последнее имя файла без расширения
    return os.path.splitext(os.path.basename(path))[0].lower()


class Case:
    def __init__(self):
        self.results: dict[tuple[str, ...], list[float]] = defaultdict(list)
        self.counts: dict[tuple[str, ...], int] = defaultdict(int)

    def add_answer(self, answer: Answer):
        test = answer["test"]
        response = answer["response"]

        frames = [frame for frame in test["frames"] if "choices" in frame]
        frames = sorted(frames, key=lambda frame: frame["path"])
        key = tuple(frame["path"] for frame in frames)

        if len(frames) != 1:  # frame
            if key not in self.results:
                self.results[key] = [0.0 for _ in key]
                self.counts[key] = 0
            if response == "Не знаю":
                for idx in range(len(frames)):
                    self.results[key][idx] += 0.5
            else:
                answer_found = False
                for idx, frame in enumerate(frames):
                    if response in frame["choices"]:
                        self.results[key][idx] += 1
                        answer_found = True
                        break

                if not answer_found:
                    print(
                        f"Warning: '{response}' не найден в choices ни одного кадра в кейсе '{answer['case_name']}'"
                    )
        else:
            if key not in self.results:
                self.results[key] = list(range(len(frames[0]["choices"])))
                self.counts[key] = 0
            self.results[key][int(response) - 1] += 1

        self.counts[key] += 1  # ⬅️ увеличиваем количество голосов на пару

    def get_rows(self):
        for key in self.results:
            yield key, self.results[key], self.counts[
                key
            ]  # теперь count — это int

    def get_total(self):
        return sum(sum(result) for result in self.results.values())


def build_confusion_and_draw_tables(case: Case):
    win_matrix = defaultdict(lambda: defaultdict(int))
    draw_counts = defaultdict(lambda: defaultdict(int))
    all_methods = set()

    for paths, scores, _ in case.get_rows():
        methods = [get_method_name(p) for p in paths]
        all_methods.update(methods)

        if len(methods) != len(scores):
            continue  # безопасность

        # сравнение всех попарно
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                s1, s2 = scores[i], scores[j]
                if s1 > s2:
                    win_matrix[m1][m2] += 1
                elif s2 > s1:
                    win_matrix[m2][m1] += 1
                else:
                    draw_counts[m1][m2] += 1
                    draw_counts[m2][m1] += 1  # симметрично

    sorted_methods = sorted(all_methods)

    # Матрица побед
    header = [""] + sorted_methods
    table = []
    for row_method in sorted_methods:
        row = [row_method]
        for col_method in sorted_methods:
            if row_method == col_method:
                row.append("-")
            else:
                row.append(win_matrix[row_method][col_method])
        table.append(row)

    print("\nМатрица конфузии (победы над другими методами):")
    print(tabulate(table, headers=header, tablefmt="grid"))

    # Таблица ничьих
    draw_table = []
    done_pairs = set()
    for m1 in sorted_methods:
        for m2 in sorted_methods:
            if m1 >= m2:
                continue
            pair = (m1, m2)
            if pair in done_pairs:
                continue
            draws = draw_counts[m1][m2]
            if draws > 0:
                draw_table.append([m1, m2, draws])
            done_pairs.add(pair)

    if draw_table:
        print("\nНичьи:")
        print(
            tabulate(
                draw_table,
                headers=["Algo1", "Algo2", "Draws"],
                tablefmt="grid",
            )
        )
    else:
        print("\nНичьи: не обнаружены.")


def save_case_results_to_csv(
    case: Case, filename: str, output_dir: str = "results_csv"
):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path1",
                "path2",
                "score1",
                "score2",
                "norm_score1",
                "norm_score2",
            ]
        )
        for paths, scores, count in case.get_rows():
            if len(paths) == 2 and len(scores) == 2:
                norm_scores = [
                    round(scores[0] / count, 3) if count > 0 else 0.0,
                    round(scores[1] / count, 3) if count > 0 else 0.0,
                ]
                writer.writerow(
                    [
                        paths[0],
                        paths[1],
                        scores[0],
                        scores[1],
                        norm_scores[0],
                        norm_scores[1],
                    ]
                )
            elif len(paths) == 1 and len(scores) == 5:
                count_scores = sum(scores)
                sum_scores = sum(
                    [cnt * val for cnt, val in enumerate(scores, 1)]
                )
                norm_score = sum_scores / count_scores
                writer.writerow(
                    [
                        paths[0],
                        None,
                        sum_scores,
                        None,
                        norm_score,
                        None,
                    ]
                )
            else:
                print(f"Пропущено (не пара): {paths} -> {scores}")

    print(f"CSV сохранён: {csv_path}")


def process_answers(json_path: str):
    cases = defaultdict(Case)

    for answer in load_answers(json_path):
        cases[answer["case_name"]].add_answer(answer)

    for case_name, case in cases.items():
        print(f"\nКейс: {case_name}")
        for paths, results, counts in case.get_rows():
            print(
                f"Пути: {paths}\nРезультаты: {results}\nКол-во голосов: {counts}"
            )
        print(f"Общее количество ответов: {case.get_total()}\n")

        build_confusion_and_draw_tables(case)

        filename = f"{case_name.replace(' ', '_').lower()}.csv"
        save_case_results_to_csv(case, filename)


def main():
    parser = ArgumentParser(description="Process answers JSONL file")
    parser.add_argument(
        "json_path",
        nargs="?",
        help="Path to answers JSONL file",
        default="answers.ldj",
    )
    args = parser.parse_args()
    process_answers(args.json_path)


if __name__ == "__main__":
    main()
