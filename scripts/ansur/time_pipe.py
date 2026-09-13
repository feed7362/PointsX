import csv
import sys
import time

sys.path.insert(0, "src")
import cv2

t0=time.perf_counter()
from webui.inference import WebuiPipeline

print(f"import {time.perf_counter()-t0:.1f}s")
rows=list(csv.DictReader(open("supabase-dump/subjects.csv",encoding="utf-8")))
r=rows[0]
t0=time.perf_counter()
p=WebuiPipeline(None,"models/yolo26-pose.pt","models/yolo12l-person-seg-extended.pt",device="cpu")
print(f"model load {time.perf_counter()-t0:.1f}s")
f=cv2.imread(r["front"]); s=cv2.imread(r["side"])
print("img", f.shape, s.shape)
for i in range(3):
    t0=time.perf_counter(); res=p.measure(f,s,float(r["height_cm"])); dt=time.perf_counter()-t0
    print(f"run{i} {dt:.2f}s  waist={res.body.waist_circumference_cm:.1f} hip={res.body.hip_circumference_cm:.1f}")
import torch, os; print("threads", torch.get_num_threads(), "cpu", os.cpu_count())
