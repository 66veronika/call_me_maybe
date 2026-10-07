import json


def load_vocabulary(path: str) -> dict[str, int]:
    """Load the model vocabulary from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            str(token): int(token_id)
            for token, token_id in data.items()
        }

    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not load vocabulary: {error}"
        ) from error
