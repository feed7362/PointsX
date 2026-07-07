# Dataset Collection Feature

A public dataset collection flow for FitMeasure AI that gathers encrypted photos and ground-truth measurements to improve the neural network classifier.

## Overview

Users can contribute to the dataset by:
1. Providing consent (18+, terms)
2. Capturing front + profile photos (reuses existing pose validation)
3. Entering 16 manual body measurements
4. Submitting encrypted data directly to Supabase

**Security:** Photos are encrypted end-to-end (libsodium sealed box) in the browser before upload. Only the offline private key can decrypt them.

## Architecture

```
Browser (dataset.html)
  ├─ Consent + DOB validation
  ├─ Camera/upload (reuses capture/* modules)
  ├─ 16 manual measurements with outlier validation
  ├─ Client-side photo encryption (libsodium sealed box)
  └─ Direct Supabase upload (supabase-js + anon key + RLS)

Supabase (FitMeasureAI project)
  ├─ dataset_submissions table (anon insert-only)
  └─ dataset-photos Storage bucket (private, anon insert-only)
```

No server endpoint required - browser talks directly to Supabase.

## Files Added

### Backend
- `scripts/gen_dataset_keypair.py` - Generate X25519 keypair
- `scripts/decrypt_dataset.py` - Decrypt collected photos offline

### Frontend
- `src/webui/static/dataset.html` - Dataset collection page
- `src/webui/static/js/dataset/app.js` - Main entry point
- `src/webui/static/js/dataset/config.js` - Supabase credentials + public key
- `src/webui/static/js/dataset/supabaseClient.js` - Encryption + upload logic
- `src/webui/static/js/dataset/measurements.js` - 16 measurement definitions + validation

### Modified Files
- `src/webui/static/index.html` - Added "Долучитись до збору датасету" button
- `src/webui/static/css/styles.css` - Dataset page styles + `.field-warn` yellow highlight
- `src/webui/static/js/capture/i18n.js` - UA/EN translations for dataset page
- `src/webui/app.py` - Added `GET /dataset.html` route
- `vercel.json` - Added dataset.html route
- `.gitignore` - Added `dataset_private_key.txt`

## Usage

### For Users (Public)

Visit: `https://your-domain.com/dataset.html`

1. Read guidelines and check consent boxes
2. Enter date of birth (must be 18+)
3. Take or upload front + profile photos
4. Enter height and sex
5. Enter 16 body measurements manually
6. Click "Надіслати дані"
   - If measurements look atypical, fields turn yellow → confirm with second click
   - Photos are encrypted in browser, then uploaded
   - Success message shows submission ID

### For Operators (Backend)

#### Initial Setup (One-time)

1. **Generate proper keypair** (replace dev keypair):
   ```bash
   pip install PyNaCl
   python scripts/gen_dataset_keypair.py
   ```
   
   - Copy the **public key** and update `src/webui/static/js/dataset/config.js`:
     ```js
     export const DATASET_PUBLIC_KEY = '<your-public-key>';
     ```
   
   - Save the **private key** securely offline (password manager, encrypted vault)
   - DO NOT commit the private key to Git

2. **Deploy updated frontend** (public key is safe to expose)

#### Retrieving Data

The Supabase MCP tools can query the data:

```bash
# List submissions
supabase-mcp list_tables --project qpxkvhmkvuqpjdeermpx

# Query submissions
supabase-mcp execute_sql --project qpxkvhmkvuqpjdeermpx \
  --query "SELECT id, created_at, age_years, height_cm, sex FROM dataset_submissions ORDER BY created_at DESC LIMIT 10"
```

Or use the Supabase dashboard:
- Tables: `dataset_submissions` (metadata + measurements)
- Storage: `dataset-photos` bucket (encrypted blobs)

#### Decrypting Photos

1. **Download encrypted photos** from Supabase Storage bucket `dataset-photos`
   
2. **Decrypt with private key**:
   ```bash
   # Single file
   python scripts/decrypt_dataset.py \
     --keyfile dataset_private_key.txt \
     --input front_1719417600_abc123.bin \
     --output front.jpg
   
   # Batch directory
   python scripts/decrypt_dataset.py \
     --keyfile dataset_private_key.txt \
     --batch downloaded_photos/ \
     --output decrypted_photos/
   ```

## Database Schema

### `public.dataset_submissions`

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| created_at | TIMESTAMPTZ | Submission timestamp |
| consent_18plus | BOOLEAN | 18+ consent (required) |
| consent_terms | BOOLEAN | Terms consent (required) |
| date_of_birth | DATE | User's DOB |
| age_years | INTEGER | Calculated age |
| height_cm | NUMERIC(5,2) | Height 100-250 cm |
| sex | TEXT | male / female / other |
| measurements | JSONB | 16 measurements {id: value} |
| front_photo_path | TEXT | Storage path to encrypted front photo |
| side_photo_path | TEXT | Storage path to encrypted side photo |
| enc_algo | TEXT | 'libsodium-sealedbox-x25519' |
| user_agent | TEXT | Browser user agent |
| app_version | TEXT | Dataset collector version |

**RLS Policy:** Anon can INSERT only (with consent checks). No SELECT/UPDATE/DELETE for anon role.

### 16 Measurements (JSONB keys)

1. `height` - Зріст (140-220 cm)
2. `neck_base_height_from_floor` - Висота точки основи шиї (110-180 cm)
3. `neck_circumference` - Обхват шиї (28-50 cm)
4. `chest_circumference` - Обхват грудей (70-150 cm)
5. `waist_circumference` - Обхват талії (55-150 cm)
6. `hip_circumference` - Обхват стегон (70-160 cm)
7. `arm_circumference_bicep` - Обхват плеча/біцепсу (20-50 cm)
8. `thigh_circumference` - Обхват стегна (35-80 cm)
9. `shoulder_width` - Ширина плечей (30-55 cm)
10. `back_width` - Ширина спини (28-50 cm)
11. `chest_width` - Ширина грудей (25-50 cm)
12. `front_length_to_waist` - Довжина переду до талії (35-65 cm)
13. `back_length_to_waist` - Довжина спини до талії (35-65 cm)
14. `sleeve_length` - Довжина рукава (50-90 cm)
15. `outer_seam` - Зовнішній шов (80-130 cm)
16. `inner_seam` - Внутрішній шов (60-100 cm)

Plausible ranges are heuristic and can be tuned in `js/dataset/measurements.js`.

## Security & Privacy

### End-to-End Encryption

- **Algorithm:** libsodium X25519 sealed box (curve25519xsalsa20poly1305)
- **Key handling:** Public key in frontend code (safe), private key offline (operator-only)
- **Plaintext lifetime:** Photos never leave the browser unencrypted (TLS + E2EE)
- **Storage:** Supabase stores only ciphertext blobs (`.bin` files)

### Row-Level Security (RLS)

- Anon role can INSERT rows (with consent validation in policy)
- Anon role CANNOT SELECT, UPDATE, or DELETE rows
- Only authenticated operators with service-role key can read data

### Metadata Privacy

- Measurements and DOB stored as plaintext (user agreed in consent)
- No PII fields (name, email, address, etc.)
- User agent logged for debugging (optional, can be removed)

## Validation & Quality Control

### Outlier Detection (Two-Stage Submit)

When user clicks "Надіслати дані":
1. Check measurements against plausible ranges
2. If outliers found → mark fields yellow + show warning → require second click
3. If user re-confirms → proceed with submission

This reduces accidental typos while allowing legitimate outliers.

### Consent Gates

- Both consent checkboxes required
- DOB must imply age ≥ 18
- Both photos required
- All 16 measurements required

RLS policy enforces consent checks at database level.

## Development Notes

### Temporary Keypair

The initial keypair in `config.js` was generated inline for development. For production:
1. Install PyNaCl: `pip install PyNaCl`
2. Run: `python scripts/gen_dataset_keypair.py`
3. Replace public key in `config.js`
4. Store private key offline securely

### Testing Locally

```bash
# Start dev server
pointsx-web

# Visit
http://localhost:8000/dataset.html
```

### Supabase Project

- **Project:** FitMeasureAI
- **ID:** qpxkvhmkvuqpjdeermpx
- **Region:** eu-west-3
- **URL:** https://qpxkvhmkvuqpjdeermpx.supabase.co

## Future Enhancements

- [ ] Add captcha/rate limiting for spam prevention
- [ ] Switch DOB storage to year-only if month/day not needed
- [ ] Add submission count dashboard for operators
- [ ] Implement measurement unit conversion (cm ↔ inches)
- [ ] Add optional ethnicity/body-type fields for diversity tracking
- [ ] Integrate with ML training pipeline (auto-labeling from Supabase)

## Troubleshooting

### "Storage upload failed: bucket not found"

- Check bucket name matches `BUCKET` in `config.js`
- Verify bucket exists in Supabase Storage

### "Database insert failed: new row violates row-level security policy"

- User did not check consent boxes
- Age < 18 based on DOB
- Check RLS policy in Supabase dashboard

### Decryption fails with "bad ciphertext"

- Wrong private key
- Corrupted download
- File was not encrypted with matching public key

### Photos not appearing after decryption

- Ensure output has correct image extension (`.jpg`, `.png`)
- Check file size (should match plaintext size, not ciphertext)
- Try opening with different image viewer

## License & Attribution

Part of the FitMeasure AI / PointsX project.
Dataset contributions are voluntary and governed by the consent terms shown to users.
