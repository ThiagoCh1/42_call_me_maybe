"""Prompt builder for the call_me_maybe project.

Builds the text prompt sent to the LLM, describing available functions
and the user request.
"""

from src.models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], user_request: str) -> str:
    """Build the prompt to send to the LLM.

    Args:
        functions: List of available function definitions.
        user_request: The natural language request from the user.

    Returns:
        The full prompt string to send to the LLM.
    """
    lines = []
    lines.append("You have access to the following functions:")
    lines.append("")

    for func in functions:
        lines.append(f"{func.name}: {func.description}")
        lines.append("  Parameters:")
        for param_name, param_def in func.parameters.items():
            lines.append(f"    - {param_name} ({param_def.type})")
        lines.append("")

    lines.append("Given the user request, determine which function to call")
    lines.append("and with what arguments.")
    lines.append("")
    lines.append(f"User request: {user_request}")

    return "\n".join(lines)
