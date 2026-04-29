# standard
import json
from enum import IntEnum, auto

# extern
from pydantic import ValidationError


class ErrorCode(IntEnum):
    """Map application failure classes to process exit codes."""

    NO_ERROR = 0
    ARGS_ERROR = auto()
    FILE_ERROR = auto()
    JSON_ERROR = auto()
    MODEL_ERROR = auto()
    RUNTIME_ERROR = auto()
    INTERRUPTED = 130


def error_code(error: Exception) -> ErrorCode:
    """Return the exit code that matches an exception."""

    message = str(error)

    if isinstance(error, json.JSONDecodeError) or "Invalid JSON" in message:
        return ErrorCode.JSON_ERROR

    if isinstance(error, (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        OSError,
    )):
        return ErrorCode.FILE_ERROR

    if (
        "LLM model is not loaded" in message
        or "CUDA out of memory" in message
        or "generating answer" in message
    ):
        return ErrorCode.MODEL_ERROR

    if isinstance(error, (ValueError, TypeError)):
        return ErrorCode.ARGS_ERROR

    return ErrorCode.RUNTIME_ERROR


def print_validation_error(error: ValidationError) -> None:
    """Print the first validation error in a compact form."""

    errors = error.errors()

    if errors:
        print(f"{type(error).__name__}: {errors[0]['msg']}")
        return

    print(f"{type(error).__name__}: {error}")
