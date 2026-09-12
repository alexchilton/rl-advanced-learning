"""Report every graded question that is still unanswered.

Three notebooks shipped with an unanswered question because the stub detector
was built from one example. This course family marks a blank at least four
ways -- `____`, an empty `""" """` query, a complete query under "# Amend the
query below", and a bare assignment like `trend = food_sales` -- and the last
two run without error, so nothing fails until a human reads the grader.

So this does not look for stubs at all. It compares each built notebook against
the untouched copy in `sources/` and lists every `q_N.check()` cell that is
byte-identical, i.e. every question left exactly as the exercise shipped it.
That is only correct when the question needs no answer, which is what the
`--thought-experiments` argument records: those question numbers are
`ThoughtExperiment` in the learntools grader and are credited by running the
cell.

    python kaggle/_tools/audit.py <course> [--thought-experiments ex2:1,3 ex4:2]

Exits non-zero if anything is unanswered, so it can gate a push.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_allowed(args):
    allowed = {}
    for item in args:
        ex, _, nums = item.partition(":")
        allowed[ex] = {int(n) for n in nums.split(",") if n}
    return allowed


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    course = sys.argv[1]
    allowed = parse_allowed(sys.argv[3:]) if "--thought-experiments" in sys.argv else {}

    problems = 0
    for built in sorted((ROOT / course).glob("ex*/*.ipynb")):
        ex = built.parent.name
        src = ROOT / course / "sources" / built.name
        if not src.exists():
            print(f"  ?? {built}: no source to compare against")
            problems += 1
            continue
        b = json.loads(built.read_text())["cells"]
        s = json.loads(src.read_text())["cells"]
        for i, (cb, cs) in enumerate(zip(b, s)):
            if cb["cell_type"] != "code":
                continue
            tb = "".join(cb["source"])
            m = re.search(r"q_(\d+)\.check\(\)", tb)
            if not m or tb != "".join(cs["source"]):
                continue
            q = int(m.group(1))
            if q in allowed.get(ex, set()):
                continue
            print(f"  UNANSWERED  {course}/{ex}  cell {i:>3}  q_{q}")
            problems += 1

    print("audit clean" if problems == 0 else f"{problems} unanswered")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
