# Tailoring Config

`tailoring_config.json` is the central data contract for garment sizing, fit ease, and pattern defaults.
It is consumed by frontend sizing and pattern-generation logic.

---

## File and Version

- Config file: `src/webui/static/data/tailoring_config.json`
- Current version: `5` (top-level `version`)

When updating semantics, increment `version` and keep migration notes with the change.

**v5 note:** Restored regional `code` strings on all **`eu_*`** and **`us_*`** grids from the canonical `EU …` / letter labels (reference: legacy `tailoring_config` v2). **`min_cm` / `max_cm` bands** stay at current project values (e.g. men v4 boundaries). **`ua_*`** grids unchanged.

**v4 note:** Men's consensus band boundaries (cm) were shifted so S/M splits align better with mainstream RTW (e.g. обхват грудей від ~94 см — це зазвичай M, не S).

---

## High-Level Structure

Top-level sections:

- `version`
- `ordinal_scales`
- `ease_profiles`
- `seam_allowances`
- `grids`
- `garments`

These sections work together:

1. `grids` map body measurements to ordinal sizes and regional labels.
2. `ordinal_scales` map ordinals to display labels (`XS`, `S`, ...).
3. `garments` choose which measurements matter and their weights.
4. `ease_profiles` and `seam_allowances` define construction/pattern defaults.

---

## `ordinal_scales`

Defines shared label arrays used by grid records:

- `top.women`
- `top.men`
- `bottom.women`
- `bottom.men`

Each grid references one scale through `scale_key`.
Ordinal index `n` must always be valid for that referenced scale.

---

## `ease_profiles`

Defines fit-dependent ease ranges per measurement key.
Each entry stores `[min, max, default]` in centimeters.

Examples of profile IDs:

- `close`
- `regular`
- `loose`
- `tailored`
- `outerwear`
- `pants`
- `skirt`

Garments pick their default via `fit_preference_default`.
Pattern logic typically uses the third number (`default`) unless explicit fit override exists.

---

## `seam_allowances`

Named seam allowance packs keyed by garment family:

- `upper`
- `lower`
- `dress`
- `full`
- `default`

Values are centimeter seam allowances by seam type (for example `side_seam`, `hem_body`, `neckline`).
Pattern code should fall back to `default` when a garment category-specific pack is unavailable.

---

## `grids`

Each grid entry defines a size-band table for one measurement axis.

Required semantics per entry:

- `id`: unique identifier
- `kind`: currently `consensus_band`
- `sex`: `female` or `male`
- `family`: `top` or `bottom`
- `measurement_key`: e.g. `chest_circumference`
- `scale_key`: key in `ordinal_scales`
- `bands`: ordered band list with:
  - `min_cm`
  - `max_cm`
  - `code` (display label — `EU …` / `EU …/…` on `eu_*` grids, `XS`…`XXL` or `S`…`XXL` on `us_*` grids; `ua_*` uses numeric codes)
  - `ordinal` (index into scale)

Band interpretation follows the sizing baseline convention:

- half-open intervals `[min_cm, max_cm)`
- `max_cm` is exclusive and belongs to the next band

Source alignment for current values is documented in:

- `src/webui/docs/sizing-baseline.md`

---

## `garments`

Each garment declares sizing and drafting behavior.

Key fields:

- `id`: stable programmatic key
- `label_uk`: user-facing Ukrainian label
- `category`: `upper` / `lower` / `dress` / `full`
- `fit_preference_default`: links to an `ease_profiles` record
- `sizing_measurements`: weighted measurements used for consensus size scoring
- `measurement_ids`: required/optional measurement set for drafting
- `audience`: `female`, `male`, or `unisex`
- `extends`: optional inheritance from another garment definition
- `svg`: icon path payload for UI rendering

### `sizing_measurements`

Each item contains:

- `id`: measurement key
- `weight`: contribution to sizing score
- `family`: expected scale family (`top` or `bottom`)

Use higher weight on primary drivers (for example chest for tops, waist/hip for bottoms).

### Inheritance via `extends`

`extends` allows a garment to reuse a base profile and override only differences.
When modifying a base garment, verify all descendants still produce expected sizing and measurement sets.

---

## Maintenance Rules

- Keep all measurement IDs consistent with emitted measurement envelopes (`/api/measure` v2+ payloads).
- Do not rename existing garment IDs unless all references are migrated.
- Keep `bands` ordered by increasing `min_cm` and aligned to ordinal progression.
- Ensure each `(sex, family, measurement_key, region)` family remains contiguous with no gaps or overlaps.
- Preserve interval semantics (`[min, max)`) across all grids.

---

## Validation Checklist Before Merge

- `version` present and integer.
- Every `grid.scale_key` resolves to an existing scale.
- Every band ordinal is in-range for that scale.
- Every garment `fit_preference_default` resolves to `ease_profiles`.
- Every garment category has a matching seam allowance pack or falls back to `default`.
- Added or changed measurement keys exist in upstream measurement payloads and downstream engines.

---

## Related Docs and Data

- `src/webui/docs/sizing-baseline.md` - source rationale for consensus band edges
- `src/webui/static/data/measurement-envelope.schema.json` - expected measurement payload contract
- `src/webui/static/js/sizeEngine.js` - consensus sizing runtime consumer
- `src/webui/static/js/patternEngine.js` - pattern/ease/seam runtime consumer
