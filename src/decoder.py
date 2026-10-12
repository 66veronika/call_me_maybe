import json
import re

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition
from src.vocab import TokenTable


def encode_text(
    model: Small_LLM_Model,
    text: str,
) -> list[int]:
    """Encode text into a flat list of token IDs."""
    encoded = model.encode(text)
    token_ids = encoded.squeeze(0).tolist()

    if isinstance(token_ids, int):
        return [token_ids]

    return [int(token_id) for token_id in token_ids]


def choose_candidate(
    model: Small_LLM_Model,
    context_ids: list[int],
    candidates: dict[str, list[int]],
) -> str:
    """Let the LLM choose one candidate token path."""
    if not candidates:
        raise ValueError("No candidates were provided.")

    active = candidates.copy()
    generated: list[int] = []

    while active:
        if len(active) == 1:
            return next(iter(active))

        position = len(generated)

        allowed = {
            sequence[position]
            for sequence in active.values()
            if position < len(sequence)
        }

        if not allowed:
            raise RuntimeError(
                "No valid candidate token remains."
            )

        if len(allowed) == 1:
            next_token = next(iter(allowed))
        else:
            logits = model.get_logits_from_input_ids(
                context_ids + generated
            )

            next_token = max(
                allowed,
                key=lambda token_id: logits[token_id],
            )

        generated.append(next_token)

        active = {
            name: sequence
            for name, sequence in active.items()
            if sequence[:len(generated)] == generated
        }

    raise RuntimeError("Could not choose a candidate.")


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt used for function selection."""
    lines = [
        "Choose the best function for the user request.",
        "Return only the function name.",
        "",
        f"User request: {user_prompt}",
        "",
        "Available functions:",
    ]

    for function in functions:
        parameters = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        )

        lines.append(
            f"- {function.name}({parameters}): "
            f"{function.description}"
        )

    lines.extend(["", "Function:"])

    return "\n".join(lines)


def choose_function_name(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Choose the function using constrained token paths."""
    if not functions:
        raise ValueError("No functions were provided.")

    prompt = build_function_prompt(
        user_prompt,
        functions,
    )

    context_ids = encode_text(model, prompt)

    candidates = {
        function.name: encode_text(
            model,
            f" {function.name}\n",
        )
        for function in functions
    }

    return choose_candidate(
        model,
        context_ids,
        candidates,
    )


def normalized_number(text: str) -> str:
    """Remove permitted leading whitespace from a number."""
    return text.lstrip(" ")


def is_number_prefix(text: str) -> bool:
    """Return whether text can still become a JSON number."""
    value = normalized_number(text)

    if value == "":
        return True

    if any(char.isspace() for char in value):
        return False

    index = 0
    length = len(value)

    if value[index] == "-":
        index += 1

        if index == length:
            return True

    if value[index] == "0":
        index += 1

        if index < length and value[index].isdigit():
            return False

    elif value[index] in "123456789":
        index += 1

        while index < length and value[index].isdigit():
            index += 1

    else:
        return False

    if index == length:
        return True

    if value[index] == ".":
        index += 1

        if index == length:
            return True

        if not value[index].isdigit():
            return False

        while index < length and value[index].isdigit():
            index += 1

    if index == length:
        return True

    if value[index] in "eE":
        index += 1

        if index == length:
            return True

        if value[index] in "+-":
            index += 1

            if index == length:
                return True

        if not value[index].isdigit():
            return False

        while index < length and value[index].isdigit():
            index += 1

    return index == length


def is_complete_number(text: str) -> bool:
    """Return whether text is a complete JSON number."""
    value = normalized_number(text)

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False

    return (
        isinstance(parsed, (int, float))
        and not isinstance(parsed, bool)
    )


def best_stop_score(
    logits: list[float],
    table: TokenTable,
) -> float:
    """Return the highest score for a JSON delimiter token."""
    valid_ids = [
        token_id
        for token_id in table.stop_ids
        if token_id < len(logits)
    ]

    if not valid_ids:
        return float("-inf")

    return max(logits[token_id] for token_id in valid_ids)


def generate_number(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    table: TokenTable,
    max_tokens: int = 16,
) -> float:
    """Generate one JSON number using constrained decoding."""
    generated: list[int] = []
    text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            prompt_ids + generated
        )

        valid_continuations = [
            token_id
            for token_id in table.number_ids
            if token_id < len(logits)
            and is_number_prefix(
                text + table.text[token_id]
            )
        ]

        best_continuation: int | None = None

        if valid_continuations:
            best_continuation = max(
                valid_continuations,
                key=lambda token_id: logits[token_id],
            )

        if is_complete_number(text):
            stop_score = best_stop_score(
                logits,
                table,
            )

            if (
                best_continuation is None
                or stop_score > logits[best_continuation]
            ):
                return float(
                    normalized_number(text)
                )

        if best_continuation is None:
            raise RuntimeError(
                "No valid number token is available."
            )

        generated.append(best_continuation)
        text += table.text[best_continuation]

    if is_complete_number(text):
        return float(normalized_number(text))

    raise RuntimeError(
        "Number generation did not finish."
    )


def is_integer_prefix(text: str) -> bool:
    """Return whether text can still become a JSON integer."""
    value = text.lstrip(" ")

    if value in ("", "-"):
        return True

    if any(char.isspace() for char in value):
        return False

    if value.startswith("-"):
        value = value[1:]

    if value == "":
        return True

    if value == "0":
        return True

    if value.startswith("0"):
        return False

    return value.isdigit()


def is_complete_integer(text: str) -> bool:
    """Return whether text is a complete JSON integer."""
    value = text.lstrip(" ")

    return re.fullmatch(
        r"-?(0|[1-9][0-9]*)",
        value,
    ) is not None


def generate_integer(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    table: TokenTable,
    max_tokens: int = 16,
) -> int:
    """Generate one JSON integer."""
    generated: list[int] = []
    text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            prompt_ids + generated
        )

        valid_continuations = [
            token_id
            for token_id in table.integer_ids
            if token_id < len(logits)
            and is_integer_prefix(
                text + table.text[token_id]
            )
        ]

        best_continuation: int | None = None

        if valid_continuations:
            best_continuation = max(
                valid_continuations,
                key=lambda token_id: logits[token_id],
            )

        if is_complete_integer(text):
            stop_score = best_stop_score(
                logits,
                table,
            )

            if (
                best_continuation is None
                or stop_score > logits[best_continuation]
            ):
                return int(text.lstrip(" "))

        if best_continuation is None:
            raise RuntimeError(
                "No valid integer token is available."
            )

        generated.append(best_continuation)
        text += table.text[best_continuation]

    if is_complete_integer(text):
        return int(text.lstrip(" "))

    raise RuntimeError(
        "Integer generation did not finish."
    )


def scan_string_body(
    text: str,
) -> tuple[bool, int | None]:
    """Validate a JSON string body and find its closing quote."""
    index = 0

    while index < len(text):
        char = text[index]

        if char == '"':
            return True, index

        if ord(char) < 32:
            return False, None

        if char == "\\":
            if index + 1 == len(text):
                return True, None

            next_char = text[index + 1]

            if next_char == "u":
                hex_part = text[
                    index + 2:index + 6
                ]

                if not all(
                    char in "0123456789abcdefABCDEF"
                    for char in hex_part
                ):
                    return False, None

                if len(hex_part) < 4:
                    return True, None

                index += 6
                continue

            if next_char not in '"\\/bfnrt':
                return False, None

            index += 2
            continue

        index += 1

    return True, None


def generate_string(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    table: TokenTable,
    max_tokens: int = 128,
) -> str:
    """Generate a JSON string body.

    The opening quote must already be present in the prompt.
    """
    generated: list[int] = []
    text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            prompt_ids + generated
        )

        ranked = np.argsort(
            np.asarray(logits)
        )[::-1]

        selected: int | None = None

        for raw_token_id in ranked:
            token_id = int(raw_token_id)

            if token_id >= len(table.text):
                continue

            piece = table.text[token_id]

            if not piece or "\ufffd" in piece:
                continue

            candidate = text + piece
            valid, closing_quote = scan_string_body(
                candidate
            )

            if not valid:
                continue

            if closing_quote is not None:
                body = candidate[:closing_quote]

                try:
                    value = json.loads(
                        f'"{body}"'
                    )
                except json.JSONDecodeError:
                    continue

                return str(value)

            selected = token_id
            break

        if selected is None:
            raise RuntimeError(
                "No valid string token is available."
            )

        generated.append(selected)
        text += table.text[selected]

    raise RuntimeError(
        "String generation did not finish."
    )


def generate_boolean(
    model: Small_LLM_Model,
    prompt_ids: list[int],
) -> bool:
    """Generate a JSON boolean."""
    candidates = {
        "true": encode_text(model, "true"),
        "false": encode_text(model, "false"),
    }

    result = choose_candidate(
        model,
        prompt_ids,
        candidates,
    )

    return result == "true"


def build_parameters_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build the argument-generation prompt."""
    schema = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )

    return (
        "Extract input arguments for a function call.\n"
        "Do not execute the function.\n\n"
        "Examples:\n"
        "Request: Combine 7 and 11\n"
        "Function: fn_pair(left: number, right: number)\n"
        'JSON arguments: {"left": 7, "right": 11}\n\n'
        "Request: Echo the string 'orange'\n"
        "Function: fn_echo(text: string)\n"
        'JSON arguments: {"text": "orange"}\n\n'
        "Request: Replace digits in 'A12B' with X\n"
        "Function: fn_replace(source: string, "
        "pattern: string, replacement: string)\n"
        'JSON arguments: {"source": "A12B", '
        '"pattern": "\\\\d+", "replacement": "X"}\n\n'
        f"Request: {user_prompt}\n"
        f"Function: {function.name}({schema})\n"
        f"Description: {function.description}\n"
        "JSON arguments: "
    )


def generate_parameters(
    model: Small_LLM_Model,
    user_prompt: str,
    function: FunctionDefinition,
    table: TokenTable,
) -> dict[str, object]:
    """Generate all required parameters in one JSON context."""
    base_prompt = build_parameters_prompt(
        user_prompt,
        function,
    )

    parameters: dict[str, object] = {}
    json_prefix = "{"

    items = list(function.parameters.items())

    for index, (name, definition) in enumerate(items):
        if index > 0:
            json_prefix += ", "

        json_prefix += f"{json.dumps(name)}: "

        parameter_type = definition.type.lower()

        if parameter_type == "string":
            context = (
                base_prompt
                + json_prefix
                + '"'
            )

            value: object = generate_string(
                model,
                encode_text(model, context),
                table,
            )

        elif parameter_type == "number":
            context = base_prompt + json_prefix

            value = generate_number(
                model,
                encode_text(model, context),
                table,
            )

        elif parameter_type == "integer":
            context = base_prompt + json_prefix

            value = generate_integer(
                model,
                encode_text(model, context),
                table,
            )

        elif parameter_type == "boolean":
            context = base_prompt + json_prefix

            value = generate_boolean(
                model,
                encode_text(model, context),
            )

        else:
            raise ValueError(
                "Unsupported parameter type: "
                f"{definition.type}"
            )

        parameters[name] = value

        json_prefix += json.dumps(
            value,
            ensure_ascii=False,
        )

    complete_json = json_prefix + "}"

    try:
        parsed = json.loads(complete_json)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Generated parameters are not valid JSON."
        ) from error

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Generated parameters are not an object."
        )

    return parameters
