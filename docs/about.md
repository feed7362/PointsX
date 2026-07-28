# PointsX

Body-measurement extraction from two ordinary photos and a known height.

## What it solves

Tailors, online clothing brands, fitness apps and clinicians all need a body
measurement table — chest, waist, hip, arm length, etc. Today most people
get measured one of three ways:

1. **In person** with a tape measure — precise, but requires a fitting
   appointment.
2. **By themselves** with a tape measure at home — error-prone, inconsistent
   between attempts, embarrassing for many users.
3. **With a 3D body scanner or specialised mobile app** — accurate but
   needs dedicated hardware or a paid subscription.

PointsX is a fourth path: **two photos and a height number**. The user stands
once for a front view, once for a side view, types their height in
centimetres, and gets back a tailoring-grade measurement table within
~10 seconds (on a GPU; ~20 seconds on a CPU).

## Who it's for

- **Custom-clothing sellers** who can't ship a tailor to every customer
  but want better data than asking for chest / waist / hips alone.
- **Researchers** studying anthropometry, ergonomics, or population-level
  body shape distributions.
- **Hobbyists and fitness coaches** who want a consistent way to track
  body composition without manual remeasuring.

PointsX is currently positioned as a **scientific demo** — accuracy is good
enough to inform garment design but not as a substitute for a clinical
anthropometric assessment.

## User experience

The flow takes about a minute from opening the page to seeing results.

1. **Open the web app** in a browser (desktop or phone).
2. **Type your height** in centimetres.
3. **Capture the front view.** The page shows a live camera feed with a
   guide silhouette. A pose-detection model runs in the browser and gives
   immediate feedback when the user is correctly framed and standing in
   the recommended pose. A voice prompt (Ukrainian) reads each instruction
   aloud so the user doesn't have to keep looking at the screen.
4. **Capture the side view.** Same process, rotated 90°.
5. **Submit.** The two photos and the height go to the inference server.
6. **See results.** A measurement table appears with eleven body
   measurements, each marked as "from front view", "from side view", or
   "fused".

Everything happens through a single web page; no app to install, no special
camera. Phones, tablets, and laptops all work — the only hard requirement
is a webcam capable of full-body framing.

## How the measurements are derived

Two photos by themselves don't contain enough information. The pipeline
combines three independent signals:

### 1. Pose keypoints (where the joints are in the picture)

A pose-estimation neural network locates sixteen anatomical landmarks per
photo — top of head, neck base, shoulders, elbows, wrists, hips, knees,
ankles. These give the structural skeleton without yet saying anything
about width or volume.

### 2. Silhouette (where the body's outline is)

A segmentation network outputs a binary mask separating the person from
the background, pixel by pixel. The mask is the foundation for every
width measurement — at any given y-coordinate the algorithm can ask
"how wide is the body here?".

### 3. Calibration (pixels to centimetres)

The known height in centimetres divided by the head-to-ankle pixel
distance gives a scale factor. Every subsequent measurement is reported
in real centimetres rather than abstract pixels.

### Combining the three

Once those three pieces are in hand, individual measurements are computed
geometrically:

- **Lengths** (arm, leg, back-to-waist) come from straight-line or
  arc-length distances between named keypoints, scaled by the calibration.
- **Widths** (chest, shoulder slope) come from the silhouette extent at
  anatomical y-rows that the keypoints anchor.
- **Circumferences** (chest, waist, hip, thigh) come from the front-view
  width and the side-view depth at the same anatomical y, fed through an
  ellipse-perimeter formula — the body cross-section is approximated as
  an ellipse with those two diameters.

A per-sex bias correction is then applied to the four circumferences to
account for the systematic difference between an ellipse and a real
human cross-section (the male torso has a slightly different shape than
the female torso when projected from front and side).

## What you get back

Eleven user-facing measurements per submission, in centimetres:

| Group | Measurements |
|---|---|
| Circumferences | chest, waist, hips, thigh |
| Widths | front chest width, shoulder slope |
| Heights | neck-base height |
| Lengths | arm (shoulder to wrist), outer leg seam, inner leg seam, back to waist |

The response also includes a confidence score per measurement and a
debug overlay (silhouette + keypoints + the lines actually used) so the
user can see *why* the pipeline produced what it produced. When the
photo doesn't meet quality requirements (head cut off, arms occluded,
extreme angle), the affected measurements are simply omitted rather
than silently making up a value.

## Architecture at a glance

```
   Browser                  Inference service
┌────────────┐           ┌─────────────────────┐
│ Camera     │           │ Pose model          │
│ Live guide │  photos + │ Segmentation model  │
│ Voice tips │ ────────→ │ Calibration         │
│ Result UI  │ ←──────── │ Measurement code    │
└────────────┘ envelope  │ Bias correction     │
                         └─────────────────────┘
```

Two independent layers, each replaceable without touching the other:

- **Browser frontend.** Captures the photos, runs a lightweight pose check
  in the browser before submitting (so users don't waste a round-trip on
  a bad shot), shows results.
- **Inference service.** A small Python web service that owns the model
  weights and the measurement code. Stateless — every request is
  independent. Returns a JSON envelope with the eleven measurements,
  per-measurement confidence, and base64-encoded debug overlays.

## Tech stack (without versions)

- **Pose & segmentation models**: YOLO-family networks (small enough to
  run on a free CPU instance, ~10–20 seconds per request; ~1 second per
  request on a GPU).
- **In-browser pose check**: MediaPipe BlazePose (runs entirely client-
  side as WebAssembly), so framing feedback is instant.
- **Backend**: FastAPI in a containerised Python runtime.
- **Frontend**: vanilla ES modules, no framework — keeps the page fast
  and dependency-free.
- **Voice prompts**: server-side neural TTS with a browser-side fallback
  via the Web Speech API.

## What the project deliberately does *not* do

- **No medical claims.** The measurements are accurate enough for
  garment fitting but should not be used to compute BMI, body-fat
  percentage, or any clinical metric.
- **No live tracking.** One submission produces one snapshot. The
  pipeline doesn't run on video streams or estimate motion.

## Current accuracy ceiling

On real-photo input the measurement table is generally within ~3 cm of a
tape-measure reading for the four core circumferences, and within ~5 cm
for the lengths and widths. That's good enough for choosing a garment
size confidently from a custom size chart and for population-level
research, but not for cutting fabric directly.

Beyond a certain point, accuracy is bounded by the photo itself:
clothing tightness, camera angle, lighting, and whether the subject
stands in the recommended pose all contribute more variance than
algorithm tuning can absorb.
