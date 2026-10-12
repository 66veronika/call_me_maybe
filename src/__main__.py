from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.io_utils import load_json
from src.models import FunctionDefinition, TestPrompt
from src.pipeline import process_prompts


def main() -> None:
    """Run the function-calling program."""
    functions_data = load_json(
        "data/input/functions_definition.json"
    )
    tests_data = load_json(
        "data/input/function_calling_tests.json"
    )

    if functions_data is None or tests_data is None:
        return

    try:
        functions = [
            FunctionDefinition.model_validate(item)
            for item in functions_data
        ]

        tests = [
            TestPrompt.model_validate(item)
            for item in tests_data
        ]

    except ValidationError as error:
        print(f"Validation error:\n{error}")
        return

    try:
        model = Small_LLM_Model()
    except Exception as error:
        print(f"Error loading model: {error}")
        return

    process_prompts(
        model,
        functions,
        tests,
    )


if __name__ == "__main__":
    main()