"""Present so that pytest puts the repository root on sys.path.

Without it, `from envs import GridWorld` fails when pytest is run from anywhere
other than the root, because pytest prepends the test file's own directory rather
than the project root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
