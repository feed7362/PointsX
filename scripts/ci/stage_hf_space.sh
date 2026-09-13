#!/usr/bin/env bash
# Assemble the Hugging Face Space tree from the monorepo.
#
# The Space needs Dockerfile + README (card front-matter) at its root; in the
# monorepo they live in apps/backend/. The Space is API-only, so the static UI
# (served by Vercel) is left out.
#
# usage: scripts/ci/stage_hf_space.sh <out_dir>
set -euo pipefail

out="${1:?usage: stage_hf_space.sh <out_dir>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rm -rf "$out"
mkdir -p "$out"

cp "$root/apps/backend/Dockerfile" \
   "$root/apps/backend/README.md" \
   "$root/apps/backend/.dockerignore" \
   "$root/apps/backend/.gitattributes" \
   "$root/pyproject.toml" \
   "$out/"

cp -R "$root/src" "$out/src"
rm -rf "$out/src/webui/static"
find "$out/src" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "staged Space tree in $out:"
(cd "$out" && find . -maxdepth 3 -not -path './src/*/*' | sort)
