import json
from argparse import ArgumentParser
from collections import defaultdict
from typing import Iterable, TypedDict
from tabulate import tabulate
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
    with open(json_path, encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def get_method_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


class Case:
    def __init__(self) -> None:
        self.results: dict[tuple[str, ...], list[float]] = defaultdict(list)

    def add_answer(self, answer: Answer) -> None:
        test = answer["test"]
        response = answer["response"]

        frames = [frame for frame in test["frames"] if "choices" in frame]
        frames = sorted(frames, key=lambda frame: frame["path"])
        key = tuple(frame["path"] for frame in frames)

        if key not in self.results:
            self.results[key] = [0.0 for _ in key]

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
                    f"Warning: '{response}' not found in choices of any frame "
                    f"in case '{answer['case_name']}'"
                )

    def get_rows(self):
        for key, result in self.results.items():
            yield key, result

    def get_total(self):
        return sum(sum(result) for result in self.results.values())


def build_confusion_and_draw_tables(case: Case) -> None:
    win_matrix = defaultdict(lambda: defaultdict(int))
    draw_counts = defaultdict(lambda: defaultdict(int))
    all_methods = set()

    for paths, scores in case.get_rows():
        methods = [get_method_name(p) for p in paths]
        all_methods.update(methods)

        if len(methods) != len(scores):
            continue

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

    print("\nMatrix of confusion (victory over other methods):")
    print(tabulate(table, headers=header, tablefmt="grid"))

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
        print("\nDraws: none found.")


def save_case_results_to_csv(
    case: Case, filename: str, output_dir: str = "results_csv"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path1", "path2", "score1", "score2"])
        for paths, results in case.get_rows():
            if len(paths) == 2 and len(results) == 2:
                writer.writerow([paths[0], paths[1], results[0], results[1]])
            else:
                print(f"Пропущено (не пара): {paths} -> {results}")

    print(f"CSV saved: {csv_path}")


def process_answers(json_path: str):
    cases = defaultdict(Case)

    for answer in load_answers(json_path):
        cases[answer["case_name"]].add_answer(answer)

    for case_name, case in cases.items():
        print(f"\nCase: {case_name}")
        for paths, results in case.get_rows():
            print(f"Paths: {paths}\nResults: {results}")
        print(f"Total number of responses: {case.get_total()}\n")

        build_confusion_and_draw_tables(case)

        filename = f"{case_name.replace(' ', '_').lower()}.csv"
        save_case_results_to_csv(case, filename)


def main():
    parser = ArgumentParser(description="Process answers JSONL file")
    parser.add_argument(
        "json_path", type=str, help="Path to answers JSONL file"
    )
    args = parser.parse_args()
    process_answers(args.json_path)


if __name__ == "__main__":
    main()
