"""Supabase access and photo decryption for the dataset viewer."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from supabase import Client, create_client

from decrypt import decrypt_bytes, load_private_key_from_file
from labels import MEASUREMENT_LABELS, MEASUREMENT_ORDER

APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")
load_dotenv(APP_DIR.parent / ".env")


class DatasetRepository:
    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "https://qpxkvhmkvuqpjdeermpx.supabase.co")
        service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not service_key:
            raise RuntimeError(
                "SUPABASE_SERVICE_ROLE_KEY is not set. Copy .env.example to .env and fill in values."
            )

        keyfile = Path(
            os.environ.get("DATASET_PRIVATE_KEY_FILE", str(APP_DIR.parent / "dataset_private_key.txt"))
        ).expanduser()
        if not keyfile.is_absolute():
            keyfile = (APP_DIR / keyfile).resolve()
        if not keyfile.is_file():
            raise RuntimeError(f"Private key file not found: {keyfile}")

        self._bucket = os.environ.get("STORAGE_BUCKET", "dataset-photos")
        self._client: Client = create_client(url, service_key)
        self._private_key = load_private_key_from_file(keyfile)
        self._derived_public_b64 = self._public_from_private(self._private_key)
        self._expected_public_b64 = self._read_expected_public_key()

    @staticmethod
    def _public_from_private(private_key) -> str:
        import base64

        return base64.b64encode(bytes(private_key.public_key)).decode()

    @staticmethod
    def _read_expected_public_key() -> str | None:
        import os
        import re

        if os.environ.get("DATASET_PUBLIC_KEY"):
            return os.environ["DATASET_PUBLIC_KEY"].strip()

        config_js = APP_DIR.parent / "src/webui/static/js/dataset/config.js"
        if not config_js.is_file():
            return None
        match = re.search(r"DATASET_PUBLIC_KEY\s*=\s*'([^']+)'", config_js.read_text())
        return match.group(1) if match else None

    @property
    def keys_match_config(self) -> bool:
        if not self._expected_public_b64:
            return True
        return self._derived_public_b64 == self._expected_public_b64

    def list_submissions(self) -> list[dict]:
        result = (
            self._client.table("dataset_submissions")
            .select("id, created_at, age_years, height_cm, sex")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    def get_submission(self, submission_id: str) -> dict:
        sid = str(UUID(submission_id))
        result = self._client.table("dataset_submissions").select("*").eq("id", sid).execute()
        if not result.data:
            raise LookupError("Submission not found")
        return _row_to_detail(result.data[0])

    def get_photo_bytes(self, submission_id: str, which: str) -> bytes:
        if which not in ("front", "side"):
            raise ValueError("Photo must be 'front' or 'side'")

        sid = str(UUID(submission_id))
        result = (
            self._client.table("dataset_submissions")
            .select("front_photo_path, side_photo_path")
            .eq("id", sid)
            .execute()
        )
        if not result.data:
            raise LookupError("Submission not found")

        row = result.data[0]
        path = row["front_photo_path"] if which == "front" else row["side_photo_path"]
        return _download_and_decrypt(self._client, self._bucket, path, self._private_key)

    def delete_submission(self, submission_id: str) -> None:
        sid = str(UUID(submission_id))
        result = (
            self._client.table("dataset_submissions")
            .select("front_photo_path, side_photo_path")
            .eq("id", sid)
            .execute()
        )
        if not result.data:
            raise LookupError("Submission not found")

        row = result.data[0]
        paths = [p for p in (row.get("front_photo_path"), row.get("side_photo_path")) if p]
        if paths:
            self._client.storage.from_(self._bucket).remove(paths)

        delete_result = self._client.table("dataset_submissions").delete().eq("id", sid).execute()
        _ = delete_result


def _row_to_detail(row: dict) -> dict:
    measurements = row.get("measurements") or {}
    ordered_measurements = [
        {
            "id": key,
            "label": MEASUREMENT_LABELS.get(key, key),
            "value": measurements.get(key),
            "unit": "cm",
        }
        for key in MEASUREMENT_ORDER
        if key in measurements
    ]
    for key, value in measurements.items():
        if key not in MEASUREMENT_ORDER:
            ordered_measurements.append(
                {"id": key, "label": MEASUREMENT_LABELS.get(key, key), "value": value, "unit": "cm"}
            )

    return {
        "id": row["id"],
        "created_at": row.get("created_at"),
        "consent_18plus": row.get("consent_18plus"),
        "consent_terms": row.get("consent_terms"),
        "date_of_birth": row.get("date_of_birth"),
        "age_years": row.get("age_years"),
        "height_cm": row.get("height_cm"),
        "sex": row.get("sex"),
        "measurements": ordered_measurements,
        "front_photo_path": row.get("front_photo_path"),
        "side_photo_path": row.get("side_photo_path"),
        "enc_algo": row.get("enc_algo"),
        "user_agent": row.get("user_agent"),
        "app_version": row.get("app_version"),
    }


def _download_and_decrypt(client: Client, bucket: str, path: str, private_key) -> bytes:
    if not path:
        raise LookupError("Photo path missing")
    blob = client.storage.from_(bucket).download(path)
    if not blob:
        raise LookupError("Photo not found in storage")
    try:
        return decrypt_bytes(blob, private_key)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. "
            "Your private key may not match the public key used when this photo was encrypted. "
            "Run: python scripts/verify_dataset_keys.py"
        ) from exc
