from llm_sdk import Small_LLM_Model

from src.decoder import choose_function_name
from src.models import FunctionDefinition, TestPrompt


def process_prompts(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    tests: list[TestPrompt],
) -> None:
    """Process all test prompts."""
    for test in tests:
        function_name = choose_function_name(
            model,
            test.prompt,
            functions,
        )

        print(
            f"{test.prompt} -> {function_name}"
        )