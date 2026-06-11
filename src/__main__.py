"""Entry point for the call_me_maybe project.

Usage:
    uv run python -m src \
        --functions_definition data/input/functions_definition.json \
        --input data/input/function_calling_tests.json \
        --output data/output/function_calls.json
"""

import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.decoder import ConstrainedDecoder
from src.models import FunctionCall, FunctionDefinition, InputPrompt
from src.prompt import build_prompt


def load_json(path: str) -> list[dict]:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a list of dicts.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained LLM decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main pipeline: load inputs, run function calling, write output."""
    args = parse_args()

    # Load and validate function definitions
    raw_functions = load_json(args.functions_definition)
    try:
        functions = [FunctionDefinition(**f) for f in raw_functions]
    except Exception as e:
        print(f"Error: invalid functions definition: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(functions)} function(s).")

    # Load and validate input prompts
    raw_prompts = load_json(args.input)
    try:
        prompts = [InputPrompt(**p) for p in raw_prompts]
    except Exception as e:
        print(f"Error: invalid input prompts: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(prompts)} prompt(s).")

    # Load model and vocabulary once
    print("Loading model... (this may take a while)")
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Run constrained decoding pipeline
    results: list[FunctionCall] = []
    for prompt in prompts:
        print(f"Processing: {prompt.prompt}")

        prompt_text = build_prompt(functions, prompt.prompt)
        input_ids = model.encode(prompt_text)[0].tolist()

        decoder = ConstrainedDecoder(model, functions, vocab, user_prompt=prompt.prompt)
        generated = decoder.generate(input_ids)

        print(f"  -> Generated: {repr(generated)}")
        try:
            data = json.loads(generated)
            result = FunctionCall(
                prompt=prompt.prompt,
                name=data["name"],
                parameters=data["parameters"],
            )
            results.append(result)
            print(f"  -> {result.name}({result.parameters})")
        except Exception as e:
            print(f"  -> Error parsing output: {e}", file=sys.stderr)

    # Write output — outside the loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
