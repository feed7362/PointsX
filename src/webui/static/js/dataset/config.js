/**
 * Dataset collection configuration
 * 
 * Supabase connection and encryption keys for the dataset submission flow.
 * 
 * SECURITY NOTES:
 * - SUPABASE_ANON_KEY is safe to expose (public/anon role with RLS)
 * - DATASET_PUBLIC_KEY is the libsodium X25519 public key for sealed box encryption
 * - Private key MUST be kept offline and never committed
 */

// Supabase project: FitMeasureAI (qpxkvhmkvuqpjdeermpx)
export const SUPABASE_URL = 'https://qpxkvhmkvuqpjdeermpx.supabase.co';
export const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFweGt2aG1rdnVxcGpkZWVybXB4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI0NjY3NDEsImV4cCI6MjA5ODA0Mjc0MX0.oYhah7P-99-Zzx6eTYXNKwCJuFvukZbW2rpbiHUPBnE';

// Storage bucket for encrypted photos
export const BUCKET = 'dataset-photos';

// libsodium X25519 public key for sealed box encryption (base64)
// Photos are encrypted client-side before upload; only the offline private key can decrypt
export const DATASET_PUBLIC_KEY = 'HxB6+jdtcSdOHKc6e/YmIH4MlQUjlJtrkwGY7sF3WW8=';

// App version for metadata tracking
export const APP_VERSION = 'v1.0';

/**
 * NOTE: This keypair was generated inline for development.
 * For production, regenerate a proper X25519 keypair:
 * 
 *   pip install PyNaCl
 *   python scripts/gen_dataset_keypair.py
 * 
 * Then update DATASET_PUBLIC_KEY above with the output.
 * Keep the private key offline and secure!
 */
