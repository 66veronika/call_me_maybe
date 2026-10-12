import sys

from llm_sdk import Small_LLM_Model

from src.decoder import (
    choose_function_name,
    generate_parameters,
)
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptItem,
)
from src.vocab import TokenTable


def process_prompts(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[PromptItem],
    table: TokenTable,
) -> list[FunctionCallResult]:
    """Process every input prompt."""
    by_name = {
        function.name: function
        for function in functions
    }

    results: list[FunctionCallResult] = []
    failures = 0

    for item in prompts:
        try:
            function_name = choose_function_name(
                model,
                item.prompt,
                functions,
            )

            function = by_name[function_name]

            parameters = generate_parameters(
                model,
                item.prompt,
                function,
                table,
            )

            result = FunctionCallResult(
                prompt=item.prompt,
                name=function_name,
                parameters=parameters,
            )

            results.append(result)

            print(
                f"{item.prompt} -> "
                f"{function_name} {parameters}"
            )

        except Exception as error:
            failures += 1

            print(
                f"Error processing "
                f"'{item.prompt}': {error}",
                file=sys.stderr,
            )

    if failures:
        raise RuntimeError(
            f"{failures} prompt(s) failed. "
            "Output file was not written."
        )

    return results
