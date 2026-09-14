"""Fail if .vercelignore would drop a file the Vercel deployment needs.

Vercel applies .vercelignore (gitignore syntax) at upload time, so a too-broad
pattern silently 404s files that exist in the `vercel` branch. That happened with
an unanchored `dataset`, which removed src/webui/static/js/dataset/.

usage: python scripts/ci/check_vercelignore.py <staged_vercel_tree>
       (run with `pathspec` available, e.g. `uv run --no-project --with pathspec ...`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pathspec

tree = Path(sys.argv[1]).resolve()
spec = pathspec.PathSpec.from_lines("gitwildmatch", (tree / ".vercelignore").read_text(encoding="utf-8").splitlines())

# Everything under these prefixes is served or imported by the deployment.
REQUIRED_PREFIXES = ("api/", "src/webui/static/", "src/webui/", "src/pointsx/")
REQUIRED_FILES = {"vercel.json", "requirements.txt"}

dropped = []
for path in sorted(p for p in tree.rglob("*") if p.is_file()):
    rel = path.relative_to(tree).as_posix()
    if "__pycache__" in rel or rel.endswith((".pyc", ".pyo")):
        continue
    needed = rel in REQUIRED_FILES or rel.startswith(REQUIRED_PREFIXES)
    if needed and spec.match_file(rel):
        dropped.append(rel)

if dropped:
    print(f".vercelignore would drop {len(dropped)} deployable file(s):", file=sys.stderr)
    for rel in dropped[:50]:
        print(f"  {rel}", file=sys.stderr)
    sys.exit(1)
print(f".vercelignore ok: no deployable file excluded ({tree})")
