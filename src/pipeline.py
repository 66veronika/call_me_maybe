from llm_sdk import Small_LLM_Model

from src.decoder import (
    choose_function_name,
    generate_parameters,
)
from src.models import FunctionDefinition, TestPrompt


def get_function_definition(
    function_name: str,
    functions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Find a function definition by name."""
    for function in functions:
        if function.name == function_name:
            return function

    raise ValueError(
        f"Function definition not found: {function_name}"
    )


def process_prompts(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    tests: list[TestPrompt],
) -> None:
    """Process all prompts."""
    for test in tests:
        try:
            function_name = choose_function_name(
                model,
                test.prompt,
                functions,
            )

            function = get_function_definition(
                function_name,
                functions,
            )

            parameters = generate_parameters(
                model,
                test.prompt,
                function,
            )

            print(f"\nPROMPT: {test.prompt}")
            print(f"FUNCTION: {function.name}")
            print(f"PARAMETERS: {parameters}")

        except (RuntimeError, ValueError) as error:
            print(
                f"Error processing '{test.prompt}': {error}"
            )