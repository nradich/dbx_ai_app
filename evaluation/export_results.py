"""Export portable measured predictions; exclude workspace identities and operational paths."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("full_plane_tests/results.json"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    portable = {key: result[key] for key in ("model_uri", "manifest_sha256", "classes", "metrics", "summary", "predictions", "scope")}
    args.output.write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
