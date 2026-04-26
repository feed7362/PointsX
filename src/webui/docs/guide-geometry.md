# Guide Geometry

`guideGeometry.js` controls the silhouette shown during capture and the tracked guide box that follows the user.  
It also defines how guide geometry is persisted and imported/exported.

---

## File and Runtime Ownership

- Runtime module: `src/webui/static/js/guideGeometry.js`
- Optional bootstrap payload: `src/webui/static/data/guide-geometry.json`
- Browser storage key: `pointsx_guide_v1`

The module exports a mutable `guideGeom` object initialized from `DEFAULT_GUIDE`.  
Most callers should rely on this live binding instead of copying geometry.

---

## Geometry Payload Shape (version 1)

Top-level fields:

- `version`: must be `1`
- `headAnchorFront`, `headAnchorProfile`: normalized anchor points for nose alignment
- `footAnchorFront`, `footAnchorProfile`: normalized anchor points for foot alignment
- `guideFrame`: tracking and sizing coefficients
- `frontPts`: closed normalized polygon for front pose
- `profilePts`: closed normalized polygon for side pose
- `profileArmPts`: open polyline for side-pose arm guide

All points are normalized `[x, y]` with expected range `[0..1]`.

---

## Core Coordinate Utilities

- `videoNormToPreviewLocal(nx, ny, cw, ch, vw, vh)`  
  Maps MediaPipe/video normalized points into preview pixels using `object-fit: cover` logic.
- `guidePtsToRefSvgPoints(pts, vbW = 100, vbH = 160)`  
  Converts normalized points into SVG point lists for reference overlays.

These are the foundation for keeping drawn guides and landmarks in the same coordinate system.

---

## Guide Box Sizing

### Baseline box

`computeGuideBox(cssW, cssH, heightCmStr, geom)` computes a default box from:

- `frameHeightFromCm()` interpolation between:
  - `fhHeightMinCm` / `fhHeightMaxCm`
  - `fhAtMin` / `fhAtMax`
- horizontal padding from `sidePadRatio`
- top offset from `bodyTopFrac`

### Tracked box

`computeGuideBoxTracked(...)` updates placement each frame and uses:

- nose anchoring (landmark `0`) when `noseVisMin` is met
- ankle stretching (landmarks `27/28`) when visibility passes `ankleVisMin`
- smoothing via `smoothFactor` + `smoothBlend`
- decay fallbacks (`decayNoLm`, `decayLowVis`) when landmarks are weak/missing
- clamping by `marginXRatio` and `marginYRatio`

The function may run in two modes:

- **Translate mode**: shifts box left/top using smoothed deltas (`dx`, `dy`)
- **Vertical-fit mode**: computes `fitHeight`, `fitTop`, `fitLeft` from nose-to-ankle span

`computeGuideBoxWithDelta()` is used to reconstruct the on-screen box using the latest smoothing state.

---

## Rendering

`drawGuideSilhouetteOnCanvas(ctx, box, currentStep, geom)` renders:

- Step 1 (front): filled and stroked `frontPts` polygon
- Step 2 (profile): filled/stroked `profilePts` plus dashed `profileArmPts`

Visual style is embedded in the renderer (stroke alpha, dash arrays, fill alpha).

---

## Persistence and Overrides

Load order:

1. `loadGuideGeometry()` tries `localStorage` payload first.
2. `loadGuideGeometryFromOptionalStaticFile()` fetches `/static/data/guide-geometry.json` only when storage is empty.

State operations:

- `saveGuideGeometry()` writes current `guideGeom` to storage
- `resetGuideGeometry()` restores defaults and clears storage
- `exportGuideGeometryJson()` returns pretty JSON string for download/copy
- `importGuideGeometryJson(str)` validates and applies payload

Validation entry point: `applyGuidePayload(o)`.

---

## Validation Rules in `applyGuidePayload`

- Rejects payloads unless `version === 1`
- Merges only finite numeric values into `guideFrame`
- Accepts anchors only when numeric `x` and `y` are present
- Accepts point arrays only when each point is exactly two finite numbers
- Requires at least:
  - `3` points for polygons (`frontPts`, `profilePts`)
  - `2` points for `profileArmPts`

This keeps malformed imports from corrupting runtime geometry.