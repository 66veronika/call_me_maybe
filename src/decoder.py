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