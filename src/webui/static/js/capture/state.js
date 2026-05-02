/**
 * Mutable capture session state (single object to avoid circular imports).
 */
export const captureState = {
  stream: null,
  step: 1,
  /** @type {Blob | null} */
  frontBlob: null,
  /** @type {Blob | null} */
  sideBlob: null,
  suspendPoseLoopAfterComplete: false,
  raf: 0,
  /** @type {any} */
  poseLandmarker: null,
  poseLoadError: null,
  /** IMAGE-mode landmarker for upload pose checks only. @type {any} */
  poseLandmarkerImage: null,
  poseImageLoadError: null,
  lastPoseCheck: 0,
  /** @type {{ ok: boolean, reason?: string }} */
  lastPoseGate: { ok: false, reason: "Завантаження…" },
  lastRawLandmarks: null,
  guideSmoothDelta: { dx: 0, dy: 0, fitHeight: null, fitTop: null, fitLeft: null },
  captureTimerIntervalId: 0,
  captureTimerRemaining: 0,
  poseStableOkSinceMs: null,
  autoPoseCountdownIntervalId: 0,
  awaitingCaptureBlob: false,
  /** @type {any | null} */
  tailoringCatalog: null,
  /** @type {any | null} MeasurementEnvelope v2 from /api/measure/mock */
  lastMockResponse: null,
  selectedGarmentId: "shirt",
  /** Fit preference forwarded to the pattern engine: "slim" | "regular" | "relaxed" */
  fitPreference: "regular",
  lastSpokenPoseMsg: "",
  lastSpokenAtMs: 0,
  /** @type {SpeechSynthesisVoice | null | undefined} */
  cachedUkVoice: undefined,
  voicesChangeHooked: false,
  sizeTabsWired: false,
  /** Debug: accept file uploads without MediaPipe pose gate (checkbox + localStorage). */
  debugSkipUploadPoseGate: false,
};
