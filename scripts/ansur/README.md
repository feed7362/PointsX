# ANSUR II experiments (docs/accuracy-approaches-survey.md §2)

Data: `ANSUR II FEMALE Public.csv` / `ANSUR II MALE Public.csv` — US Army 2012, cleared for
unlimited public release (shippable). Mirror:
https://github.com/senihberkay/US-Army-ANSUR-II (raw/master/). Units: mm (÷10), weightkg ×0.1.

- `ell.py`   — Ramanujan ellipse bias + per-site/sex scale k
- `reg.py`   — 5-fold CV: H | H+W+BMI | ellipse | ellipse+H+W
- `noise.py` — sensitivity to width noise and clothing inflation
- `time_pipe.py` — CPU timing of the production pipeline (run from repo root)
- `bodym_weight.py` — BodyM (EVAL-ONLY, CC BY-NC): girth ± height/weight on real masks.
  Numbers are benchmark-only, never shippable.

Scripts expect `a.csv` (female) / `m.csv` (male) in cwd.
