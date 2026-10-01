import json

from typing import Any

# class 
def load_json(path: str) -> Any:
    """Load and return data from a JSON file."""

    try:
        with open(path, "r", encoding="utf-8",) as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None

    except json.JSONDecodeError:
        print(f"Error: invalid JSON in: {path}")
        return None