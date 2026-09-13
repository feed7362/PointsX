# PointsX — `vercel` deploy branch

**Generated.** This branch is the Vercel deploy tree built from `main` by
`.github/workflows/deploy.yml` (`scripts/ci/stage_vercel.sh`). Do not develop here —
edit `main`, let CI pass, and the deploy job commits the new tree on top of this branch.

Manual fallback (second method, kept on purpose): any commit you push here deploys as-is
(`git push origin <commit>:vercel`). CI never force-pushes, so your commit stays in the
history; the next deploy from `main` overlays it. Hotfixes made here must also land on `main`
or they are lost at the next deploy.
