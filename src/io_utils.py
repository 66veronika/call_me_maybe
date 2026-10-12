import json
from pathlib import Path
from typing import Any

from src.models import FunctionCallResult


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None

    except json.JSONDecodeError:
        print(f"Error: invalid JSON in: {path}")
        return None


def write_results(
    path: str,
    results: list[FunctionCallResult],
) -> None:
    """Write generated function calls to JSON."""
    output_path = Path(path)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                [
                    result.model_dump()
                    for result in results
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

    except OSError as error:
        raise RuntimeError(
            f"Could not write output file: {error}"
        ) from error
