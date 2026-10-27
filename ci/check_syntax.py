"""Check tracked Python/JavaScript syntax and JSON/YAML parsing without imports.

This does not execute notebooks, train a model, or validate Databricks SQL.
Requires git, Node.js, and PyYAML. Run from any directory in this checkout.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8')
    checked, errors = Counter(), []
    for relative in filter(None, tracked.split('\0')):
        path = ROOT / relative
        suffix = path.suffix.lower()
        if suffix not in {'.py', '.js', '.mjs', '.json', '.yaml', '.yml'}:
            continue
        try:
            if suffix == '.py':
                compile(path.read_text(encoding='utf-8-sig'), relative, 'exec')
            elif suffix == '.json':
                json.loads(path.read_text(encoding='utf-8-sig'))
            elif suffix in {'.yaml', '.yml'}:
                yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            else:
                subprocess.run(['node', '--check', str(path)], check=True, cwd=ROOT)
            checked[suffix] += 1
        except (OSError, ValueError, SyntaxError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
            errors.append(f'{relative}: {exc}')
    for suffix, count in sorted(checked.items()):
        print(f'Checked {count} {suffix} files')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
