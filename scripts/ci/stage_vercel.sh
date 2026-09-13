#!/usr/bin/env bash
# Assemble the Vercel deploy tree (contents of the `vercel` branch) from the monorepo.
#
# Only what api/index.py needs: the FastAPI app in proxy mode + the static UI.
# pyproject.toml / uv.lock are deliberately absent — with them present Vercel
# tries to resolve torch and the build fails (see the old deploy-vercel.sh).
#
# usage: scripts/ci/stage_vercel.sh <out_dir>
set -euo pipefail

out="${1:?usage: stage_vercel.sh <out_dir>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rm -rf "$out"
mkdir -p "$out"

cp "$root/vercel.json" "$root/requirements.txt" "$root/.vercelignore" "$out/"
cp -R "$root/api" "$out/api"
mkdir -p "$out/src"
cp -R "$root/src/pointsx" "$out/src/pointsx"
cp -R "$root/src/webui" "$out/src/webui"
find "$out" -type d -name __pycache__ -prune -exec rm -rf {} +
# training / synthetic tooling is never imported by the Vercel function
rm -rf "$out/src/pointsx/synthetic" "$out/src/pointsx/regression" \
       "$out/src/pointsx/train_pose.py" "$out/src/pointsx/train_seg.py" \
       "$out/src/pointsx/convert_pose.py" "$out/src/pointsx/verify_labels.py" \
       "$out/src/pointsx/eval.py"

cat > "$out/README.md" <<'MD'
# PointsX — `vercel` deploy branch

**Generated.** This branch is the Vercel deploy tree built from `main` by
`.github/workflows/deploy.yml` (`scripts/ci/stage_vercel.sh`). Do not develop here —
edit `main`, let CI pass, and the deploy job commits the new tree on top of this branch.

Manual fallback (second method, kept on purpose): any commit you push here deploys as-is
(`git push origin <commit>:vercel`). CI never force-pushes, so your commit stays in the
history; the next deploy from `main` overlays it. Hotfixes made here must also land on `main`
or they are lost at the next deploy.
MD

echo "staged Vercel tree in $out:"
(cd "$out" && find . -maxdepth 2 | sort)
