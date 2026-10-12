from typing import Any

from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Describe one function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: str


class ReturnDefinition(BaseModel):
    """Describe a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """Describe one callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptItem(BaseModel):
    """Represent one natural-language input prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent one generated function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
