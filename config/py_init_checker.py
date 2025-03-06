import os
from typing import List

from tap import Tap


class ArgumentParser(Tap):
    directory: str
    exclude: List[str] = ["venv"]


def check_init_files(directory: str, exclude: List[str]) -> None:
    problem_directories = []

    for root, _, files in os.walk(directory):
        if any(excluded_directory in root for excluded_directory in exclude):
            continue
        if any(file.endswith(".py") for file in files):
            if "__init__.py" not in files:
                problem_directories.append(root)

    if problem_directories:
        raise ValueError(
            f'The following directories do not contain "__init__.py" file. Please add them. {problem_directories}'
        )
    print(f"Directory {directory} has all the necessary __init__.py files.")


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    check_init_files(args.directory, args.exclude)
