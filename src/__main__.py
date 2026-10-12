import argparse

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.io_utils import load_json, write_results
from src.models import FunctionDefinition, PromptItem
from src.pipeline import process_prompts
from src.vocab import build_token_table


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/output/"
            "function_calling_results.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the function-calling pipeline."""
    args = parse_arguments()

    functions_data = load_json(
        args.functions_definition
    )

    prompts_data = load_json(
        args.input
    )

    if functions_data is None or prompts_data is None:
        return

    if not isinstance(functions_data, list):
        print("Error: function definitions must be a list.")
        return

    if not isinstance(prompts_data, list):
        print("Error: input prompts must be a list.")
        return

    try:
        functions = [
            FunctionDefinition.model_validate(item)
            for item in functions_data
        ]

        prompts = [
            PromptItem.model_validate(item)
            for item in prompts_data
        ]

    except ValidationError as error:
        print(f"Validation error:\n{error}")
        return

    try:
        model = Small_LLM_Model()

        table = build_token_table(
            model,
            model.get_path_to_vocab_file(),
        )

        results = process_prompts(
            model,
            functions,
            prompts,
            table,
        )

        write_results(
            args.output,
            results,
        )

        print(
            f"Results written to {args.output}"
        )

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
