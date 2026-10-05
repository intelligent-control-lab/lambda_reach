"""Check a frozen policy/value bundle against explicit held-out accuracy gates."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safety_value.monitoring.qualification import qualify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--limits", type=Path, help="Optional JSON with predeclared limit overrides")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    limits = json.loads(args.limits.read_text()) if args.limits else None
    result = qualify_bundle(args.bundle, limits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
