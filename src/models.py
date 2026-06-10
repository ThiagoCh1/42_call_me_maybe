"""Pydantic models for the call_me_maybe project.

These models represent the data structures used throughout the pipeline:
- Input: function definitions and natural language prompts
- Output: resolved function calls with typed arguments
"""

from typing import Any
from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """Represents a single parameter of a function, as defined
    in functions_definition.json."""

    type: str


class FunctionDefinition(BaseModel):
    """Represents a callable function with its signature, as
    defined in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class InputPrompt(BaseModel):
    """Represents a single natural language
    prompt from function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """Represents the resolved output: which function
    to call and with what arguments."""

    prompt: str
    name: str
    parameters: dict[str, Any]
