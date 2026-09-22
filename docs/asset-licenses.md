# Asset licenses

Every asset that reaches a rendered image must be listed here with a license that allows
**commercial** use. This is the same discipline as the eval datasets: BodyM / SHAPY-HBW are
research-only and never touch training or shipped weights (`eval/bodym.py` carries the guard).

| Asset | Source | License | Use |
|---|---|---|---|
| `assets/hdri/*.hdr` (5) | Poly Haven | CC0 | environment lighting |
| `assets/clothing/` | — | — | empty; to be filled with CC0 garments (step 3) |

## Removed 2026-09-20/22

| What | Why |
|---|---|
| `assets/textures/smplitex/` (106 MB) | SMPL-derived UV textures: wrong topology for the new body model and the license was never cleared |
| `data/synthetic-pose/` (6.4 GB) + `.zip` (2.4 GB), `data/synthetic-seg/` (172 MB) | generated before the 2026-07-18 GT fix — 131/200 sampled bodies fail `pointsx.gt_sanity`, waist mean 111.6 cm vs ~86 real, hips 129.4 vs ~102; SMPL-X-derived (non-commercial) |

## Body model (decision pending, step 2)

- **MPFB2 / MakeHuman** — add-on code GPL, generated **assets/outputs CC0**; ships CC0 clothing and skins.
- **Anny** (Naver) — Apache 2.0, smplx-compatible topology, phenotype parameters. The *Anny-One dataset* is
  non-commercial: code only.
- **SMPL-X** — non-commercial (commercial licensing via Meshcapade). Not used.

## Reference only, never training data

CLOTH3D, CLOTH4D, SynBody, AGORA, BEDLAM, BodyM, SHAPY/HBW — research licenses. Used to compare
approaches and, where the license allows, as held-out evaluation.

## Tooling

Blender 5.2.2 LTS (`Q:\SteamLibrary\steamapps\common\Blender\blender.exe`) — GPL; renders and cloth
simulation are our own output. ANSUR II (US Army, public release) — the shape prior, shippable.
