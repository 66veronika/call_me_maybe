from pydantic import ValidationError

from src.io_utils import load_json
from src.models import FunctionDefinition, TestPrompt
from llm_sdk import Small_LLM_Model


def main() -> None:
    functions_data = load_json("data/input/functions_definition.json")
    tests_data = load_json("data/input/function_calling_tests.json")

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

    print("FUNCTIONS:")
    for function in functions:
        print(function)

    print("\nTEST PROMPTS:")
    for test in tests:
        print(test)


if __name__ == "__main__":
    main()