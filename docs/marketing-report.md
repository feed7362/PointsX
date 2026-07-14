# FitMeasure AI — Market Context Report

> Sprint framing: aligning the offering with actual market demand and communicating its value proposition.

---

## 1. Problem statement

Getting accurate body measurements today is either inconvenient, error-prone, or expensive:

| Method | Friction | Cost | Accuracy |
|---|---|---|---|
| In-person tailor appointment | High (scheduling, travel) | High | Best |
| Self-measurement with tape | Low (setup) | Free | Poor (±5–10 cm typical) |
| 3D body scanner | Very high (specialist hardware) | Very high | Best |
| Specialised mobile app (e.g. MySizeID, Bodygee) | Medium (proprietary hardware requirements or paid subscription) | Medium–High | Good |
| **FitMeasure AI** | **Low (two photos + known height)** | **SaaS (subscription + pay-as-you-go)** | **Good** |

The gap FitMeasure AI fills: **tailoring-grade measurements with no hardware and no appointment**.

---

## 2. Product snapshot

**What it is:** A two-layer web service that extracts 11 body measurements from one front photo and one side photo, using the user's known height as the only additional input.

**What it produces in one session:**

| Group | Measurements |
|---|---|
| Circumferences | Chest, Waist, Hips, Thigh |
| Widths | Front chest width, Shoulder slope |
| Lengths | Arm, Outer leg seam, Inner leg seam, Back-to-waist |
| Heights | Neck-base height |

**Accuracy:** target ~±3 cm, sufficient for garment-size selection. Note: **circumferences are the hardest measurement** to recover from 2D photos (lengths are easier) — improving circumference accuracy is the main pipeline challenge.

**UX flow:** open browser → enter height → front photo with voice guidance → side photo → measurement table. Total time ~1 minute. No app install. Works on phone, tablet, laptop.

---

## 3. Market segments

### Primary: custom and made-to-measure clothing (B2B)

Custom clothing brands, Etsy tailors, white-label garment manufacturers, and print-on-demand apparel platforms all face the same problem: asking the customer for 10+ measurements produces bad data (most people don't own a tape measure and measure themselves badly).

**Pain:** returns and remakes caused by wrong measurements cost the custom apparel industry an estimated 10–15% of gross revenue.

**How FitMeasure AI helps:** embed the measurement widget (or API) into the checkout flow. The customer takes two photos; the brand receives a structured JSON envelope with 11 measurements. No tailor visit, no customer error, no size chart ambiguity.

**Entry point:** white-label API integration or embeddable widget (`<iframe>` or JS SDK).

---

## 4. Value proposition (by segment)

| Segment | Core promise | Key differentiator vs. status quo |
|---|---|---|
| Custom apparel brands | Cut returns caused by wrong self-measurements | No app install, no hardware — embeds in existing checkout |
| Made-to-measure tailors | Instant digital measurement profile per client | Replaces the manual tape session |

---

## 5. Competitive landscape

| Competitor | Approach | Gap FitMeasure AI exploits |
|---|---|---|
| MySizeID | Mobile app + ML on single front photo | Requires dedicated app install; single-view limits depth estimate |
| Bodygee | Smartphone + turntable / 3-photo rig | Hardware dependency and subscription cost |
| Nettelo | App with ~20-photo capture | High capture friction |
| 3DLOOK / YourFit | Enterprise SaaS, CV pipeline | Priced out of reach for small brands and individuals |
| Manual size chart + self-input | Zero friction for brand | ±5–10 cm error → high return rate |

FitMeasure AI's structural advantages:
- **No install, no hardware.** Pure browser capture.
- **Open / embeddable.** REST API and iframe-friendly, not locked to a marketplace.
- **Transparent accuracy.** Confidence scores and a debug overlay are returned with every result — the product doesn't over-claim.
- **Deployable on free infrastructure.** CPU inference on a free-tier container (~20 s/request) keeps marginal cost near zero at demo/pilot scale.

---

## 6. Go-to-market sequence

### Stage 0 — Scientific demo (current)
Position: university program showcase, proof-of-concept for potential partners.
Goal: validate accuracy claims with real users, collect ~50 labeled measurement pairs for error-rate analysis.
Distribution: public Vercel URL shared via program showcase.

### Stage 1 — Pilot with one custom-clothing brand
Target: a mid-size Etsy or independent DTC brand with 200–1 000 orders/month.
Offer: free integration for 3 months in exchange for anonymized return-rate data before vs. after.
Deliverable: embeddable JS widget + webhook that pushes measurement JSON to the brand's order management system.
Success metric: return rate for custom orders drops ≥ 15% vs. the same brand's historical baseline.

### Stage 2 — API-first SaaS launch
Pricing model: **subscription + pay-as-you-go** (per-measurement) + white-label tier (custom domain, logo, monthly flat rate). **No freemium.**
Distribution: ProductHunt launch, direct outreach to Shopify app ecosystem, GitHub README targeting developer-first apparel startups.

### Stage 3 — Vertical expansion
Options (ranked by demand signal):
1. Academic / ergonomics licensing.
2. Additional apparel verticals (uniforms, sportswear, made-to-measure).

---

## 7. Key risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Accuracy insufficient for high-stakes garment cutting | Medium | High | Position explicitly as "size selection", not "pattern cutting". Publish accuracy numbers up front. |
| User unwilling to share full-body photos | Medium | High | No photo storage (process and discard), GDPR-compliant by design; make this explicit in the UI. |
| Incumbent (3DLOOK / MySizeID) replicates browser-capture approach | High | Medium | Win on underserved niches and a simpler, non-bloated, privacy-first product; deepen per-brand size mapping and API integration. (No reliance on storing measurement history — conflicts with our privacy stance.) |
| Free-tier infrastructure (HF Spaces CPU) too slow for B2B SLA | High | Medium | GPU upgrade on HF Spaces or migration to a 1-core cloud VM at ~$5/month when first paying customer onboards. |

---

## 8. Positioning statement

**For** online clothing brands, made-to-measure tailors, and body-measurement researchers  
**who** need accurate body measurements without a tape measure or specialist hardware,  
**FitMeasure AI** is a web-based measurement service  
**that** extracts 11 garment-grade body measurements from two standard photos in under a minute.  
**Unlike** mobile apps that require installation or enterprise scanners that require hardware,  
**FitMeasure AI** works in any browser, embeds into any checkout flow via a REST API, and is transparent about its accuracy on every result.

---

## 9. Open questions for the sprint

1. **Pricing calibration.** At what price per measurement does the API become self-funding at expected pilot volume? (Needs cost-per-request number from infrastructure.)
2. **Privacy positioning.** Is "we discard photos after processing" enough, or does the target B2B segment require on-premise deployment for GDPR compliance?
3. **Accuracy floor for apparel.** What ±cm tolerance do the target brand partners actually require? (Currently ±3 cm — confirm this is acceptable for the size-chart use case, not just "good in theory".)
4. **Privacy / E2E encryption.** E2E encryption is mandatory for the EU market.
