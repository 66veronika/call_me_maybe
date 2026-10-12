import json
from typing import Any

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel


class TokenTable(BaseModel):
    """Cache useful decoded vocabulary information."""

    text: list[str]
    number_ids: list[int]
    integer_ids: list[int]
    stop_ids: list[int]


def load_vocabulary(path: str) -> dict[str, int]:
    """Load the model vocabulary."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data: Any = json.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Vocabulary file not found: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid vocabulary JSON: {path}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError("Vocabulary must be a JSON object.")

    vocabulary: dict[str, int] = {}

    for token, token_id in data.items():
        if not isinstance(token, str):
            raise RuntimeError("Invalid vocabulary token.")

        if not isinstance(token_id, int):
            raise RuntimeError("Invalid vocabulary token ID.")

        vocabulary[token] = token_id

    return vocabulary


def is_stop_piece(text: str) -> bool:
    """Return whether a token begins a JSON value delimiter."""
    if text.startswith("\n"):
        return True

    return text.lstrip(" ").startswith((",", "}"))


def build_token_table(
    model: Small_LLM_Model,
    vocab_path: str,
) -> TokenTable:
    """Decode the vocabulary once and cache useful token groups."""
    vocabulary = load_vocabulary(vocab_path)

    if not vocabulary:
        raise RuntimeError("Vocabulary is empty.")

    max_id = max(vocabulary.values())
    text = [""] * (max_id + 1)

    for token_id in set(vocabulary.values()):
        text[token_id] = model.decode([token_id])

    number_chars = set("0123456789.-+eE ")
    integer_chars = set("0123456789- ")

    number_ids = [
        token_id
        for token_id, piece in enumerate(text)
        if piece
        and not piece.isspace()
        and set(piece) <= number_chars
    ]

    integer_ids = [
        token_id
        for token_id, piece in enumerate(text)
        if piece
        and not piece.isspace()
        and set(piece) <= integer_chars
    ]

    stop_ids = [
        token_id
        for token_id, piece in enumerate(text)
        if piece and is_stop_piece(piece)
    ]

    return TokenTable(
        text=text,
        number_ids=number_ids,
        integer_ids=integer_ids,
        stop_ids=stop_ids,
    )
