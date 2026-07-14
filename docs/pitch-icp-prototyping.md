# FitMeasure AI — Pitch, ICP & Prototyping Brief

Body measurements from 2 photos (front + side) + known height → sizes in cm.
Online-first. **B2B primary**, B2C secondary.

---

## Ideal Customer Profile (ICP)

### Who is the customer?
**Primary (B2B):** online-first apparel businesses that lose money on wrong sizes.
- Local online clothing stores / DTC brands.
- Tailors & ateliers (made-to-measure).
- Other online retailers / marketplaces needing a fit feature.

**Secondary (B2C):** the end customer who buys online and doesn't know their size.

> **Not primary:** offline shops. They already have a shop assistant who measures.
> Selling "reduce your staffing" to offline retail is highly competitive and complex —
> a **future option, not the entry point**.

### What is their problem?
- End users **measure themselves wrong** (don't know how, or won't spend the minutes).
- Wrong size → **returns, refunds, staff time re-making garments, wasted material**.
- Existing fit tools are slow, app/hardware-heavy, or enterprise-priced.

### What do we offer?
A web service that takes **2 photos + height** and returns body measurements + the
recommended size in **seconds**, removing the end-customer human factor.

### What is the product/service?
- **SaaS API** (front + side photo → measurements in cm + size recommendation).
- **Custom size charts per business** — size charts differ not just by country but by
  every brand/manufacturer; each business maps our measurements onto its own chart.
- Runs in **resource-constrained environments** → low hosting cost.

### How does it solve the problem?
- For the **business**: fewer wrong-size returns → less refund cost, less staff time,
  less wasted fabric/material.
- For the **end customer**: instant, no need to know how to self-measure, better UX.
- Cuts measurement from **minutes → seconds** with meaningful accuracy.

### Why better than existing solutions?
- **Faster** than most fit tools.
- **Higher throughput on constrained hardware** → cheaper hosting for the business.
- **No app, no 3D scanner, no special hardware** — just photos in a browser.
- **Removes the end-user human factor** instead of trusting users to measure right.
- **We attack where big, well-funded incumbents fail** *(Vera).* They can out-spend and
  out-compute us, so we don't fight on raw scale. We win on the gaps they leave:
  underserved niches (accessibility), and a simple, non-bloated solution instead of a
  heavy enterprise platform.

### Unique value
> **We move the measurement off the untrained end user and onto an instant,
> hardware-free engine — so online apparel businesses stop paying for wrong-size
> returns, wasted material, and staff rework.**

### Accessibility — kept as an edge case for now
People with disabilities feel the worst pain with offline fitting, so they're a
valuable future segment. But it's a **much harder problem to solve**, so for now we
treat it as an **edge case** and cover the standard customer first.

### Trust & privacy as a marketing pillar *(Ralf)*
End customers must upload **full-body photos** — adoption depends on trust. E2E
encryption + GDPR aren't just technical checkboxes; **privacy and trust are a core
marketing message.** No B2C trust → no B2C uploads → no value for the B2B client.

---

## Pitch (Problem → Next steps)

### Problem
Online apparel returns run **20–40%**, and **fit is the #1 reason**. The root cause:
end users self-measure badly or skip it. That's refund cost, staff rework time, and
wasted material for the business.

### Solution
**2 photos + height → measurements + size, in seconds.** No app, no hardware.
We remove the end-customer measurement error before the order is placed.

### Product / Service
- Input: front + side photo + height → output: body measurements in cm + recommended size.
- Computer vision from a photo; known height calibrates pixels to centimetres.
- Delivery: **SaaS API**. Each business maps results onto **its own size chart**.

### Traction *(honest — no invented numbers)*
- Working **MVP pipeline** (photo + height → measurements).
- **Feedback session** with potential **B2B and B2C** customers on a Fashion Forum
  (a couple of months ago).
- Live landing: **fitmeasureai.pp.ua**.
- Currently **polishing the core pipeline** for accuracy across clothing types and heights.

> Target accuracy ±3 cm and reduced-return numbers are **goals to validate in pilot**,
> not yet proven.

### Team
6 students of Khmelnytskyi National University, supervised by **Maryna Molchanova**
(PhD, Senior Lecturer):
- **Kostiantyn Lianskorunskyi** — Team Lead · PM · ML
- **Maksim Shurypa** — ML · Developer
- **Yaroslav Kazmirchuk** — Frontend · ML
- **Maksym Mykytiuk** — Data Analytics · ML
- **Daria Kryva** — UI/UX · Frontend
- **Vladyslav Murava** — Data Analytics · ML

### Next steps
1. Collect real ground-truth measurements → validate accuracy.
2. Closed pilot with 1–2 local brands / ateliers (early autumn), pitched as an
   **A/B test** *(Ralf)*: retailer serves traditional size charts to one customer group
   and FitMeasure AI to another → **definitively prove** lower returns / higher sales
   vs. the baseline.
3. Public **MVP launch mid–late autumn 2026** (peak autumn/winter wardrobe season).
4. Expand garment types + harden API.

---

## Business model
- **SaaS-first.**
- **Subscription + pay-as-you-go.**
- **No freemium.**
- **Custom size charts** — each brand/manufacturer has its own chart; we map
  measurements onto it.

---

## Prototyping brief

### Current prototype state
End-to-end pipeline works: front + side photo + height → measurements. Landing live.
**Accuracy not yet calibrated on real people** — current focus is polishing the core
pipeline for varied clothing and heights.

### What we want from prototyping
- Prove/disprove target accuracy **±3 cm** on real bodies.
- Find where the pipeline breaks: poses, clothing, lighting, body types, height range.
- Validate the UX hypothesis: can users take correct photos without instructions?

### Measurable validation hypotheses *(Vera)*
- **Time:** FitMeasure AI cuts time-to-measurement by **≥50%** vs. manual methods.
- **Willingness to pay:** **≥50%** of Ukrainian made-to-measure clothing manufacturers
  will pay **≥€100/month** for the service.
- **Returns (pilot A/B):** measurable drop in wrong-size returns vs. size-chart baseline.

### How we validate — observe, don't just ask *(Vera)*
Don't only interview users — **physically observe** customers and tailors taking manual
measurements in real time. Reveals the real friction and "jobs to be done" that people
won't articulate in an interview.

### Next steps in developing the idea
Ground-truth collection → calibration → B2B pilot → public MVP in autumn.

### Ideas from the prototyping session

**Security — E2E encryption (EU-mandatory).**
Photos are sensitive personal data. For the future **EU market**, treat
**end-to-end encryption as mandatory** — needed for GDPR trust and B2B procurement.

**Accessibility — edge case for now.**
Partial testing on the Fashion Forum:
- ✅ **One-legged person** — works.
- ❌ **Wheelchair user** — does not work.

Disabled users are a valuable future segment, but a **much harder problem to solve**, so
we keep them as an **edge case** and prove the standard customer first.

### Top questions for experts
1. **Validation data** — fastest ethical way to collect real ground-truth body
   measurements without our own lab (crowdsourcing, atelier partnership, open datasets)?
2. **Accuracy bar** — what error (±cm) is actually good enough for a business to trust
   our size recommendation?
3. **Go-to-market** — best way into local online brands/ateliers with a pilot
   (return-rate proof, free test, or other)?
