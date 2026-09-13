"""BodyM (eval-only, CC BY-NC): does adding height/weight to the width->girth map help on REAL masks?
Fit on train (n=300), evaluate on testB (n=400). Numbers are benchmark-only, non-shippable."""
import json
import sys

import numpy as np

sys.path.insert(0, "eval"); sys.path.insert(0, "src")
import bodym as B

s3 = B._client()
def collect(split, n):
    meas, hwg, photo = B.load_split(s3, split)
    subs = [s for s in meas if s in photo and s in hwg][:n or None]
    rows = []
    for i, sid in enumerate(subs, 1):
        pred = B.subject_pred(s3, split, sid, hwg, photo)
        h = float(hwg[sid]["height_cm"]); w = float(hwg[sid]["weight_kg"]); sex = hwg[sid]["gender"]
        r = {"sex": sex, "h": h, "w": w, "bmi": w / (h / 100) ** 2}
        for m in B.CIRCUMFERENCES:
            v = pred.get(m); r[m] = (v, float(meas[sid][m]))
        rows.append(r)
        if i % 100 == 0: print(f"  {split} {i}/{len(subs)}", flush=True)
    return rows
tr = collect("train", 300); te = collect("testB", 400)
json.dump({"train": tr, "testB": te}, open(sys.argv[1], "w"))
def X(rows, m, kind):
    out = []; y = []
    for r in rows:
        v, gt = r[m]
        if v is None: continue
        circ, fw, sw = v
        feats = {"scale": [circ],
                 "girth": [fw, sw, fw * sw],
                 "girth+hw": [fw, sw, fw * sw, r["h"], r["w"], r["bmi"]],
                 "hw": [r["h"], r["w"], r["bmi"]]}[kind]
        out.append(feats); y.append(gt)
    return np.array(out), np.array(y)
print("\nBodyM testB MAE (cm), fit on train n=300, per sex. scale=median-ratio(B); girth=bilinear(D); +hw adds height/weight/BMI; hw=no silhouette")
print(f"{'sex':7s}{'meas':7s} {'raw':>6s} {'scale':>6s} {'girth':>6s} {'girth+hw':>9s} {'hw-only':>8s}")
for sex in ("female", "male"):
    trs = [r for r in tr if r["sex"] == sex]; tes = [r for r in te if r["sex"] == sex]
    for m in B.CIRCUMFERENCES:
        res = {}
        Xtr, ytr = X(trs, m, "scale"); Xte, yte = X(tes, m, "scale")
        res["raw"] = np.abs(Xte[:, 0] - yte).mean()
        k = np.median(ytr / Xtr[:, 0]); res["scale"] = np.abs(Xte[:, 0] * k - yte).mean()
        for kind in ("girth", "girth+hw", "hw"):
            Xtr, ytr = X(trs, m, kind); Xte, yte = X(tes, m, kind)
            A = np.c_[np.ones(len(Xtr)), Xtr]; w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            res[kind] = np.abs(np.c_[np.ones(len(Xte)), Xte] @ w - yte).mean()
        print(f"{sex:7s}{m.split('_')[0]:7s} {res['raw']:6.2f} {res['scale']:6.2f} {res['girth']:6.2f} {res['girth+hw']:9.2f} {res['hw']:8.2f}   n_te={len(yte)}")
