"""Allow ``python -m lumen`` as an alias for ``python -m lumen.indexer``."""

from lumen.indexer import main

if __name__ == "__main__":
    main()
