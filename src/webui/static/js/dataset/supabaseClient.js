/**
 * Supabase client and dataset submission functions
 * 
 * Handles end-to-end encryption of photos and direct submission to Supabase.
 * Uses libsodium sealed box (X25519) for photo encryption in the browser.
 */

import {
  SUPABASE_URL,
  SUPABASE_ANON_KEY,
  BUCKET,
  DATASET_PUBLIC_KEY,
  APP_VERSION
} from './config.js';

// Lazy-load dependencies from CDN
let supabase = null;
let sodium = null;

/**
 * Initialize Supabase client (lazy)
 */
/**
 * Import the first specifier that loads.
 *
 * Both dependencies used to be pulled from esm.sh alone, so a CDN hiccup — or an
 * ad-blocker / corporate proxy that filters esm.sh — broke dataset submission
 * outright with "Failed to fetch dynamically imported module". Encryption is on
 * the critical path here, so a single CDN is not an acceptable dependency.
 *
 * jsdelivr's /+esm endpoint serves self-contained ES modules, which makes it a
 * genuine fallback rather than a mirror of the same infrastructure.
 *
 * @param {string[]} specifiers - URLs to try, in order
 * @param {string} label - human name used in the final error
 */
async function importWithFallback(specifiers, label) {
  const failures = [];
  for (const url of specifiers) {
    try {
      return await import(/* @vite-ignore */ url);
    } catch (err) {
      failures.push(`${new URL(url).host}: ${err?.message ?? err}`);
      console.warn(`[dataset] ${label} failed from ${url}`, err);
    }
  }
  throw new Error(
    `Не вдалося завантажити ${label}. Перевірте підключення до мережі або ` +
    `блокувальник реклами. Деталі: ${failures.join(' | ')}`,
  );
}

async function getSupabase() {
  if (supabase) return supabase;

  const { createClient } = await importWithFallback([
    'https://esm.sh/@supabase/supabase-js@2',
    'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm',
    'https://unpkg.com/@supabase/supabase-js@2/dist/module/index.js',
  ], 'Supabase client');
  supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return supabase;
}

/**
 * Initialize libsodium (lazy)
 */
async function getSodium() {
  if (sodium) return sodium;

  const libsodiumModule = await importWithFallback([
    'https://esm.sh/libsodium-wrappers@0.7.13',
    'https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm',
    'https://unpkg.com/libsodium-wrappers@0.7.13/dist/modules/libsodium-wrappers.js',
  ], 'бібліотеку шифрування (libsodium)');
  const lib = libsodiumModule.default ?? libsodiumModule;
  await lib.ready;
  sodium = lib;
  return sodium;
}

/**
 * Encrypt a photo blob using libsodium sealed box
 * 
 * @param {Blob} photoBlob - The photo to encrypt (JPEG/PNG/WebP)
 * @returns {Promise<{ciphertext: Uint8Array, originalSize: number}>}
 */
export async function encryptPhoto(photoBlob) {
  const sodium = await getSodium();
  
  // Convert blob to Uint8Array
  const arrayBuffer = await photoBlob.arrayBuffer();
  const plaintext = new Uint8Array(arrayBuffer);
  
  // Decode public key from standard base64 (PyNaCl / libsodium ORIGINAL variant)
  const publicKey = sodium.from_base64(
    DATASET_PUBLIC_KEY,
    sodium.base64_variants.ORIGINAL
  );
  
  // Encrypt using sealed box (anonymous encryption)
  // Only the holder of the private key can decrypt
  const ciphertext = sodium.crypto_box_seal(plaintext, publicKey);
  
  return {
    ciphertext,
    originalSize: plaintext.length
  };
}

/**
 * Upload encrypted photo to Supabase Storage
 * 
 * @param {Uint8Array} ciphertext - Encrypted photo bytes
 * @param {string} filename - Filename for storage (e.g., 'front_12345.bin')
 * @returns {Promise<{path: string}>}
 */
async function uploadEncryptedPhoto(ciphertext, filename) {
  const supabase = await getSupabase();
  
  // Upload as binary blob (application/octet-stream)
  const { data, error } = await supabase.storage
    .from(BUCKET)
    .upload(filename, ciphertext, {
      contentType: 'application/octet-stream',
      upsert: false
    });
  
  if (error) {
    throw new Error(`Storage upload failed: ${error.message}`);
  }
  
  return { path: data.path };
}

/**
 * Insert metadata row into dataset_submissions table
 * 
 * @param {Object} submission - Submission data
 * @returns {Promise<{id: string}>}
 */
async function insertSubmission(submission) {
  const supabase = await getSupabase();

  // Anon role has INSERT-only RLS; .select() after insert requires a SELECT policy.
  const { error } = await supabase
    .from('dataset_submissions')
    .insert(submission);

  if (error) {
    throw new Error(`Database insert failed: ${error.message}`);
  }

  return { id: submission.id };
}

/**
 * Submit complete dataset entry
 * 
 * Encrypts both photos, uploads ciphertext to Storage, and inserts metadata.
 * 
 * @param {Object} params
 * @param {Blob} params.frontBlob - Front photo blob
 * @param {Blob} params.sideBlob - Side photo blob
 * @param {boolean} params.consent18plus
 * @param {boolean} params.consentTerms
 * @param {string} params.dateOfBirth - ISO date string (YYYY-MM-DD)
 * @param {number} params.ageYears
 * @param {number} params.heightCm
 * @param {string} params.sex - 'male' | 'female' | 'other'
 * @param {Object} params.measurements - Object with 16 measurement keys
 * @returns {Promise<{id: string, frontPath: string, sidePath: string}>}
 */
export async function uploadAndInsert({
  frontBlob,
  sideBlob,
  consent18plus,
  consentTerms,
  dateOfBirth,
  ageYears,
  heightCm,
  sex,
  measurements
}) {
  // Validate required fields
  if (!frontBlob || !sideBlob) {
    throw new Error('Both front and side photos are required');
  }
  if (!consent18plus || !consentTerms) {
    throw new Error('All consents must be checked');
  }
  if (ageYears < 18) {
    throw new Error('Must be 18 or older');
  }
  
  // Generate unique filenames using timestamp + random suffix
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  const frontFilename = `front_${timestamp}_${randomSuffix}.bin`;
  const sideFilename = `side_${timestamp}_${randomSuffix}.bin`;
  
  try {
    // Encrypt photos
    console.log('Encrypting photos...');
    const [frontEncrypted, sideEncrypted] = await Promise.all([
      encryptPhoto(frontBlob),
      encryptPhoto(sideBlob)
    ]);
    
    console.log(`Front: ${frontBlob.size} bytes -> ${frontEncrypted.ciphertext.length} bytes encrypted`);
    console.log(`Side: ${sideBlob.size} bytes -> ${sideEncrypted.ciphertext.length} bytes encrypted`);
    
    // Upload encrypted photos to Storage
    console.log('Uploading encrypted photos to Storage...');
    const [frontUpload, sideUpload] = await Promise.all([
      uploadEncryptedPhoto(frontEncrypted.ciphertext, frontFilename),
      uploadEncryptedPhoto(sideEncrypted.ciphertext, sideFilename)
    ]);
    
    // Insert metadata row
    console.log('Inserting metadata...');
    const submissionId = crypto.randomUUID();
    const submission = {
      id: submissionId,
      consent_18plus: consent18plus,
      consent_terms: consentTerms,
      date_of_birth: dateOfBirth,
      age_years: ageYears,
      height_cm: heightCm,
      sex: sex,
      measurements: measurements,
      front_photo_path: frontUpload.path,
      side_photo_path: sideUpload.path,
      enc_algo: 'libsodium-sealedbox-x25519',
      user_agent: navigator.userAgent,
      app_version: APP_VERSION
    };
    
    const { id } = await insertSubmission(submission);
    
    console.log(`✓ Dataset submission complete! ID: ${id}`);
    
    return {
      id,
      frontPath: frontUpload.path,
      sidePath: sideUpload.path
    };
    
  } catch (error) {
    console.error('Dataset submission failed:', error);
    throw error;
  }
}
