# standard
import sys
import os

# extern
import fire
from pydantic import ValidationError

# local
from .cli import CLI
from .error import ErrorCode, print_validation_error, error_code


def main() -> ErrorCode:
    """Run the command-line application."""

    os.environ.setdefault("PAGER", "cat")

    try:
        fire.Fire(CLI())

    except ValidationError as e:
        print_validation_error(e)
        return ErrorCode.ARGS_ERROR

    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return ErrorCode.INTERRUPTED

    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return error_code(e)

    return ErrorCode.NO_ERROR


if __name__ == "__main__":
    sys.exit(main())
