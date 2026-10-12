import json

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


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


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Create a prompt containing the available functions."""
    lines = [
        "Choose the best function for the user request.",
        "Return only the function name.",
        "",
        f"User request: {user_prompt}",
        "",
        "Available functions:",
    ]

    for function in functions:
        lines.append(
            f"- {function.name}: {function.description}"
        )

    lines.append("")
    lines.append("Function:")

    return "\n".join(lines)


def choose_function_name(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Choose a valid function using constrained decoding."""
    if not functions:
        raise ValueError("No functions available.")

    prompt = build_function_prompt(
        user_prompt,
        functions,
    )

    prompt_ids = encode_text(model, prompt)

    candidates = {
        function.name: encode_text(
            model,
            f" {function.name}",
        )
        for function in functions
    }

    generated_ids: list[int] = []
    active_candidates = candidates.copy()

    while active_candidates:
        logits = model.get_logits_from_input_ids(
            prompt_ids + generated_ids
        )

        position = len(generated_ids)

        allowed_tokens = {
            token_ids[position]
            for token_ids in active_candidates.values()
            if position < len(token_ids)
        }

        if not allowed_tokens:
            raise RuntimeError(
                "No valid token available."
            )

        next_token = max(
            allowed_tokens,
            key=lambda token_id: logits[token_id],
        )

        generated_ids.append(next_token)

        active_candidates = {
            name: token_ids
            for name, token_ids in active_candidates.items()
            if token_ids[:len(generated_ids)]
            == generated_ids
        }

        for name, token_ids in active_candidates.items():
            if token_ids == generated_ids:
                return name

    raise RuntimeError(
        "Could not select a function."
    )

def build_parameters_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build a prompt for generating function input arguments."""
    schema = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )

    return (
        "Extract the INPUT ARGUMENTS for a function call.\n"
        "You are NOT executing the function.\n"
        "You are ONLY identifying values that should be passed to it.\n\n"

        "Rules:\n"
        "- Never calculate or execute the function.\n"
        "- Never invent placeholder values or variable names.\n"
        "- Never output text such as 'user_input_string'.\n"
        "- Use information from the user's request.\n"
        "- Preserve literal input strings instead of transforming them.\n"
        "- A source/input string is the text the function should operate on.\n"
        "- A regex parameter is the regex pattern needed to match what "
        "the user described.\n"
        "- A replacement parameter is what matched text should be "
        "replaced with.\n"
        "- Different parameters should contain their own appropriate "
        "values.\n\n"

        f"Function: {function.name}\n"
        f"Function description: {function.description}\n"
        f"Required parameters: {schema}\n"
        f"User request: {user_prompt}\n\n"

        "JSON arguments:\n"
    )

def build_parameter_context(
    user_prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    parameter_type: str,
    extracted: dict[str, object],
) -> str:
    """Build a prompt for one parameter with previous values as context."""
    parameter_schema = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )

    previous = json.dumps(
        extracted,
        ensure_ascii=False,
    )

    return (
        "You are extracting INPUT ARGUMENTS for a function call.\n"
        "Do NOT execute the function.\n"
        "Do NOT calculate its result.\n"
        "Do NOT return a JSON object.\n"
        "Return only the value of the requested parameter.\n\n"
        f"Function: {function.name}({parameter_schema})\n"
        f"Description: {function.description}\n"
        f"User request: {user_prompt}\n\n"
        f"Already extracted parameters: {previous}\n"
        f"Parameter to extract now: {parameter_name}\n"
        f"Required type: {parameter_type}\n\n"
        "Use the parameter name to understand which value "
        "belongs to it.\n"
        "Do not repeat a previous parameter unless the user "
        "actually provided the same value for both.\n\n"
        f"Value for {parameter_name}:\n"
    )

def generate_value(
    model: Small_LLM_Model,
    context: str,
    parameter_type: str,
) -> object:
    """Generate one value while preserving the full JSON context."""
    context_ids = encode_text(model, context)

    if parameter_type == "number":
        return generate_number(model, context_ids)

    if parameter_type == "string":
        return generate_string(model, context_ids)

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )

def generate_parameters(
    model: Small_LLM_Model,
    user_prompt: str,
    function: FunctionDefinition,
) -> dict[str, object]:
    """Generate all parameters using one continuous JSON context."""
    base_prompt = build_parameters_prompt(
        user_prompt,
        function,
    )

    parameters: dict[str, object] = {}
    json_prefix = "{"

    parameter_items = list(function.parameters.items())

    for index, (name, definition) in enumerate(parameter_items):
        if index > 0:
            json_prefix += ", "

        json_prefix += f"{json.dumps(name)}: "

        context = (
            base_prompt
            + json_prefix
            + f"\n\nGenerate ONLY the value for parameter '{name}'."
        )
        value = generate_value(
            model,
            context,
            definition.type,
        )

        parameters[name] = value

        json_prefix += json.dumps(
            value,
            ensure_ascii=False,
        )

    json_prefix += "}"

    return parameters

def is_number_prefix(text: str) -> bool:
    """Return whether text can still become a valid JSON number."""
    if text == "":
        return True

    index = 0
    length = len(text)

    if text[index] == "-":
        index += 1

        if index == length:
            return True

    if text[index] == "0":
        index += 1

        if index < length and text[index].isdigit():
            return False

    elif text[index] in "123456789":
        index += 1

        while index < length and text[index].isdigit():
            index += 1

    else:
        return False

    if index == length:
        return True

    if text[index] == ".":
        index += 1

        if index == length:
            return True

        if not text[index].isdigit():
            return False

        while index < length and text[index].isdigit():
            index += 1

    if index == length:
        return True

    if text[index] in "eE":
        index += 1

        if index == length:
            return True

        if text[index] in "+-":
            index += 1

            if index == length:
                return True

        if not text[index].isdigit():
            return False

        while index < length and text[index].isdigit():
            index += 1

    return index == length


def is_complete_number(text: str) -> bool:
    """Return whether text is a complete JSON number."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def generate_number(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    max_tokens: int = 16,
) -> int | float:
    """Generate a JSON number using constrained decoding."""
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        input_ids = prompt_ids + generated_ids
        logits = model.get_logits_from_input_ids(input_ids)

        ranked_tokens = np.argsort(logits)[::-1]

        current_text = (
            model.decode(generated_ids)
            if generated_ids
            else ""
        )

        selected_token: int | None = None

        for token_id_raw in ranked_tokens:
            token_id = int(token_id_raw)

            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)

            if candidate_text == current_text:
                continue

            # Newline means the model wants to finish the number.
            if "\n" in candidate_text:
                value_text, _, remainder = candidate_text.partition("\n")

                if (
                    remainder == ""
                    and is_complete_number(value_text)
                ):
                    return json.loads(value_text)

                continue

            if is_number_prefix(candidate_text):
                selected_token = token_id
                break

        if selected_token is None:
            raise RuntimeError(
                "No valid number token could be generated."
            )

        generated_ids.append(selected_token)

    text = model.decode(generated_ids)

    if is_complete_number(text):
        return json.loads(text)

    raise RuntimeError("Number generation did not finish.")

def is_string_prefix(text: str) -> bool:
    """Return whether text can become a valid JSON string."""
    if text == "":
        return True

    if not text.startswith('"'):
        return False

    escaped = False

    for index, char in enumerate(text[1:], start=1):
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            return index == len(text) - 1

        if ord(char) < 32:
            return False

    return True


def is_complete_string(text: str) -> bool:
    """Return whether text is a complete JSON string."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False

    return isinstance(value, str)


def generate_string(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    max_tokens: int = 128,
) -> str:
    """Generate a JSON string using constrained decoding."""
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        input_ids = prompt_ids + generated_ids
        logits = model.get_logits_from_input_ids(input_ids)

        ranked_tokens = np.argsort(logits)[::-1]

        current_text = (
            model.decode(generated_ids)
            if generated_ids
            else ""
        )

        selected_token: int | None = None

        for token_id_raw in ranked_tokens:
            token_id = int(token_id_raw)

            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)

            if candidate_text == current_text:
                continue

            if not is_string_prefix(candidate_text):
                continue

            selected_token = token_id

            if is_complete_string(candidate_text):
                return json.loads(candidate_text)

            break

        if selected_token is None:
            raise RuntimeError(
                "No valid string token could be generated."
            )

        generated_ids.append(selected_token)

    raise RuntimeError("String generation did not finish.")