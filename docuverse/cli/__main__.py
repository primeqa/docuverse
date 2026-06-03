"""Allow ``python -m docuverse.cli`` as an alias for the console script."""
import sys

from docuverse.cli import main

if __name__ == "__main__":
    sys.exit(main())
