# Sizing Baseline — Reconciled Brand Size Tables

Source brands: Zara, H&M, Uniqlo, Mango, Nike (activewear cross-check).
All measurements in cm (body, not garment). Data captured April 2026.
Every band in `tailoring_config.json` v2 traces back to this document.

---

## Methodology

1. Download each brand's published body-measurement size guide (not garment measurements).
2. For each (sex, garment-category, measurement), note the per-brand band edges.
3. Where brands diverge by ≤ 2 cm at an edge the reconciled midpoint is used.
4. Where brands diverge by > 2 cm, the most conservative (smaller-band) edge is used so the system errs toward recommending the larger size.
5. All bands are half-open **[min, max)** — a measurement equal to `max_cm` falls into the next band.

---

## Women's Tops (Shirts, T-shirts, Blouses, Dresses, Jackets)

### Chest Circumference — ordinals 0–5

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | XS | 78 – 82 | 36 | EU 34 | XS |
| 1 | S | 82 – 86 | 38 | EU 36 | S |
| 2 | M | 86 – 90 | 40 | EU 38 | M |
| 3 | L | 90 – 94 | 42 | EU 40 | L |
| 4 | XL | 94 – 100 | 44 | EU 42 | XL |
| 5 | XXL | 100 – 116 | 46/48 | EU 44 | XXL |

Brand divergence notes:
- H&M M: 86–91, Zara M: 87–91, Uniqlo M: 86–90, Mango M: 86–91 → reconciled 86–90 (tight).
- Nike XL starts at 95 (activewear) vs 94 (RTW) → RTW convention used.

### Waist Circumference (bodice waist) — ordinals 0–5

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | XS | 58 – 62 | 36 | EU 34 | XS |
| 1 | S | 62 – 66 | 38 | EU 36 | S |
| 2 | M | 66 – 70 | 40 | EU 38 | M |
| 3 | L | 70 – 76 | 42 | EU 40 | L |
| 4 | XL | 76 – 84 | 44 | EU 42 | XL |
| 5 | XXL | 84 – 96 | 46/48 | EU 44 | XXL |

### Hip Circumference (relevant for fitted tops, jackets, dresses) — ordinals 0–5

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | XS | 84 – 88 | 36 | EU 34 | XS |
| 1 | S | 88 – 92 | 38 | EU 36 | S |
| 2 | M | 92 – 96 | 40 | EU 38 | M |
| 3 | L | 96 – 102 | 42 | EU 40 | L |
| 4 | XL | 102 – 110 | 44 | EU 42 | XL |
| 5 | XXL | 110 – 124 | 46/48 | EU 44 | XXL |

---

## Women's Bottoms (Pants, Skirts, Shorts)

### Waist Circumference — ordinals 0–5

Same band edges as women's tops waist column above (same body measurement, same scale).

### Hip Circumference — ordinals 0–5

Same band edges as women's tops hip column above.

### Thigh Circumference — ordinals 0–5

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | XS | 46 – 50 | 36 | EU 34 | XS |
| 1 | S | 50 – 54 | 38 | EU 36 | S |
| 2 | M | 54 – 58 | 40 | EU 38 | M |
| 3 | L | 58 – 64 | 42 | EU 40 | L |
| 4 | XL | 64 – 72 | 44 | EU 42 | XL |
| 5 | XXL | 72 – 84 | 46 | EU 44 | XXL |

Brand notes: Thigh bands derived from Zara and Uniqlo extended-fit guides; Nike activewear +2 cm per band for stretch (not applied here — use RTW values).

---

## Men's Tops (Shirts, T-shirts, Jackets, Coats)

### Chest Circumference — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 88 – 96 | 44/46 | EU 46 | S |
| 1 | M | 96 – 104 | 48/50 | EU 48/50 | M |
| 2 | L | 104 – 112 | 52 | EU 52 | L |
| 3 | XL | 112 – 120 | 54/56 | EU 54 | XL |
| 4 | XXL | 120 – 130 | 58 | EU 56 | XXL |

Brand notes: Uniqlo M ends at 103, H&M M ends at 104 → 104 used; Zara XL starts at 113 → conserved at 112.

### Waist Circumference (bodice) — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 76 – 84 | 44/46 | EU 46 | S |
| 1 | M | 84 – 92 | 48/50 | EU 48/50 | M |
| 2 | L | 92 – 100 | 52 | EU 52 | L |
| 3 | XL | 100 – 108 | 54/56 | EU 54 | XL |
| 4 | XXL | 108 – 118 | 58 | EU 56 | XXL |

### Hip Circumference — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 90 – 98 | 44/46 | EU 46 | S |
| 1 | M | 98 – 106 | 48/50 | EU 48/50 | M |
| 2 | L | 106 – 114 | 52 | EU 52 | L |
| 3 | XL | 114 – 122 | 54/56 | EU 54 | XL |
| 4 | XXL | 122 – 132 | 58 | EU 56 | XXL |

---

## Men's Bottoms (Pants, Jeans, Shorts)

### Waist Circumference — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 76 – 82 | 46 | EU 46 | S |
| 1 | M | 82 – 88 | 48 | EU 48 | M |
| 2 | L | 88 – 96 | 50/52 | EU 50 | L |
| 3 | XL | 96 – 104 | 54 | EU 52 | XL |
| 4 | XXL | 104 – 114 | 56 | EU 54 | XXL |

### Hip Circumference — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 90 – 98 | 46 | EU 46 | S |
| 1 | M | 98 – 106 | 48 | EU 48 | M |
| 2 | L | 106 – 114 | 50/52 | EU 50 | L |
| 3 | XL | 114 – 122 | 54 | EU 52 | XL |
| 4 | XXL | 122 – 132 | 56 | EU 54 | XXL |

### Thigh Circumference — ordinals 0–4

| Ordinal | Label | Band (cm) | UA | EU | US |
|---------|-------|-----------|-----|-----|-----|
| 0 | S | 48 – 54 | 46 | EU 46 | S |
| 1 | M | 54 – 60 | 48 | EU 48 | M |
| 2 | L | 60 – 68 | 50/52 | EU 50 | L |
| 3 | XL | 68 – 76 | 54 | EU 52 | XL |
| 4 | XXL | 76 – 86 | 56 | EU 54 | XXL |

---

## Ease Allowance Reference (wearing ease E, in cm)

Source: Müller & Sohn, Winifred Aldrich *Metric Pattern Cutting* (7th ed.), ЕМКО ЦОТШЛ.

| Fit Profile | Applied to | Chest E | Waist E | Hip E | Thigh E |
|-------------|-----------|---------|---------|-------|---------|
| close | dress, blouse, fitted knit | 4–6 | 1–2 | 2–4 | 2–4 |
| regular | shirt, tshirt, polo | 8–12 | 4–6 | 4–6 | 4–6 |
| loose | hoodie, sport_top, pajama | 14–18 | 8–10 | 8–10 | 6–10 |
| tailored | jacket, vest | 10–14 | 6–8 | 6–8 | 4–6 |
| outerwear | coat, raincoat | 16–22 | 12–14 | 10–14 | 8–12 |
| pants | pants, jeans, shorts | 0 | 2–3 | 4–6 | 4–6 |
| skirt | skirt types | 0 | 1–2 | 2–4 | 2–4 |

Default column (index 2) used by `easeFor()` in `patternEngine.js`.

---

## Pattern Formula Reference

Source: Müller & Sohn *System der Maßschneiderei*, Aldrich *Metric Pattern Cutting*.

| Formula | Expression | Notes |
|---------|-----------|-------|
| Armscye depth | C/10 + 10.5 (F) / +11.0 (M) | C = chest circumference |
| Cross-back | BW + 0.8 | BW = back_width_scapular |
| Cross-front | FW + 0.4 (F) / +0.8 (M) | FW = chest_width_front |
| Shoulder seam | SS / 2 | SS = shoulder_slope_width |
| Neck width | N/5 + 0.5 | N = neck_circumference |
| Neck depth front | neck_width + 0.5 | — |
| Neck depth back | 2.0 cm (constant) | — |
| Dart intake | (H − W) / 4 | back 55%, side 30%, front 15% |
| Hip line depth | BL/2 + 2 | BL = back_length_to_waist |
| Sleeve cap height | AD × 0.75 (F) / ×0.80 (M) | AD = armscye depth |
| Sleeve cap width | C × 0.18 + 2 | — |
| Crotch depth | H/4 + 1 (F) / +1.5 (M) | — |
| Bicep pattern | UA + E_bicep | E_bicep: slim=2, reg=4, relaxed=6 |
| Cuff pattern | wrist + 3 (button) / +1 (knit) | — |
