# standard imports
import sys

# local imports
from .cli import main


if __name__ == "__main__":
    sys.exit(main().value)
