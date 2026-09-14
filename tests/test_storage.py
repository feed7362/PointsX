"""Behaviour of the archival/storage layer (``webui.infrastructure.storage``).

Written against the former single ``webui/storage.py`` module and kept green
across the split into s3/supabase/local/crypto/common. No network: Supabase REST
and boto3 are replaced by fakes.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

STORAGE_ENV = (
    "LOCAL_DATA_DIR", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_STORAGE_BUCKET", "STORAGE_BUCKET",
    "S3_PREFIX", "SUPABASE_PREFIX", "POINTSX_ARCHIVE_ENCRYPT", "DATASET_PUBLIC_KEY", "POINTSX_S3_NO_METADATA",
    *(f"{p}{n}" for p in ("S3_", "ELK_", "R2_", "B2_", "AWS_")
      for n in ("ENDPOINT", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY", "BUCKET", "REGION", "PREFIX", "FORCE_PATH_STYLE")),
)


@pytest.fixture
def storage(monkeypatch):
    """Fresh storage module (the S3 client/config is cached per process) with a clean env."""
    for key in STORAGE_ENV:
        monkeypatch.delenv(key, raising=False)
    for name in [m for m in sys.modules if m.startswith("webui.infrastructure.storage")]:
        del sys.modules[name]
    return importlib.import_module("webui.infrastructure.storage")


ARCHIVE_ARGS = dict(
    front_bytes=b"\xff\xd8front", front_content_type="image/jpeg",
    side_bytes=b"\x89PNGside", side_content_type="image/png",
    envelope_json={"schema": "pointsx.measurement.envelope", "label": "Обхват"},
    metadata={"height_cm": "175.0", "sex": "female"},
)


# ── configuration ─────────────────────────────────────────────────────────────

def test_disabled_without_configuration(storage):
    assert storage.is_enabled() is False
    assert storage.archive_measurement("rid", **ARCHIVE_ARGS) is False
    assert storage.archive_measurement_local("rid", **ARCHIVE_ARGS) is False


def test_s3_prefix_group_selection_and_endpoint_rules(storage, monkeypatch):
    monkeypatch.setenv("ELK_ENDPOINT", " https://abc.supabase.co/ \n")
    monkeypatch.setenv("ELK_ACCESS_KEY_ID", "key\n")
    monkeypatch.setenv("ELK_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("ELK_BUCKET", "bucket")
    monkeypatch.setenv("ELK_PREFIX", "pre/")
    assert storage.is_enabled() is True
    cfg = storage.s3._load_config()
    assert cfg.endpoint == "https://abc.supabase.co/storage/v1/s3"  # bare Supabase URL auto-corrected
    assert cfg.access_key_id == "key" and cfg.region == "us-east-1" and cfg.prefix == "pre"
    assert cfg.force_path_style is True  # non-AWS → path-style

    monkeypatch.setenv("S3_ENDPOINT", "https://s3.amazonaws.com")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "b")
    monkeypatch.setenv("S3_BUCKET", "c")
    cfg = storage.s3._load_config()
    assert cfg.endpoint == "https://s3.amazonaws.com" and cfg.force_path_style is False  # S3_ wins, AWS → virtual
    monkeypatch.setenv("S3_FORCE_PATH_STYLE", "yes")
    assert storage.s3._load_config().force_path_style is True


def test_helpers(storage):
    assert storage.common._sanitise_request_id("a/b c?d-1.2_x") == "a_b_c_d-1.2_x"
    assert storage.common._sanitise_request_id("") == "anon"
    assert storage.common._sanitise_request_id("x" * 200) == "x" * 96
    assert [storage.common._extension_for(c) for c in ("image/JPEG", "image/png", "image/webp", "", None)] == [
        "jpg", "png", "webp", "bin", "bin"]


# ── local filesystem ──────────────────────────────────────────────────────────

def test_local_archive_layout(storage, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    assert storage.archive_measurement_local("req/1", **ARCHIVE_ARGS) is True
    folder = tmp_path / "measurements" / "req_1"
    assert sorted(p.name for p in folder.iterdir()) == ["envelope.json", "front.jpg", "metadata.json", "side.png"]
    assert (folder / "front.jpg").read_bytes() == ARCHIVE_ARGS["front_bytes"]
    assert json.loads((folder / "envelope.json").read_text(encoding="utf-8")) == ARCHIVE_ARGS["envelope_json"]
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8")) == ARCHIVE_ARGS["metadata"]


def test_local_model_path_lookup_order(storage, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    assert storage.local_model_path("w.pt") is None
    (tmp_path / "w.pt").write_bytes(b"root")
    assert storage.local_model_path("w.pt") == tmp_path / "w.pt"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "w.pt").write_bytes(b"preferred")
    assert storage.local_model_path("w.pt") == tmp_path / "models" / "w.pt"


def test_download_to_path_skips_existing_and_needs_config(storage, tmp_path):
    existing = tmp_path / "have.pt"
    existing.write_bytes(b"x")
    assert storage.download_to_path("models/have.pt", existing) is True
    assert storage.download_to_path("models/missing.pt", tmp_path / "missing.pt") is False


# ── Supabase Storage REST ─────────────────────────────────────────────────────

class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append({"url": req.full_url, "method": req.get_method(), "headers": dict(req.header_items()),
                      "body": req.data, "timeout": timeout})
        return _FakeResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _supabase_env(monkeypatch, **extra):
    monkeypatch.setenv("SUPABASE_URL", "https://ref.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_x")
    monkeypatch.setenv("STORAGE_BUCKET", "dataset")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_supabase_plain_upload(storage, monkeypatch):
    _supabase_env(monkeypatch, POINTSX_ARCHIVE_ENCRYPT="0", SUPABASE_PREFIX="/arch/")
    calls = _capture_urlopen(monkeypatch)
    assert storage.is_enabled() is True
    assert storage.archive_measurement("r 1", **ARCHIVE_ARGS) is True
    base = "https://ref.supabase.co/storage/v1/object/dataset/arch/measurements/r_1/"
    assert [c["url"] for c in calls] == [base + "front.jpg", base + "side.png", base + "envelope.json"]
    assert [c["headers"]["Content-type"] for c in calls] == ["image/jpeg", "image/jpeg", "application/json"]
    assert calls[0]["headers"]["Apikey"] == "sb_secret_x" and calls[0]["headers"]["Authorization"] == "Bearer sb_secret_x"
    assert calls[0]["headers"]["X-upsert"] == "true" and calls[0]["method"] == "POST" and calls[0]["timeout"] == 30
    assert calls[0]["body"] == ARCHIVE_ARGS["front_bytes"]
    assert json.loads(calls[2]["body"].decode("utf-8")) == ARCHIVE_ARGS["envelope_json"]


def test_supabase_sealed_upload(storage, monkeypatch):
    pytest.importorskip("nacl")
    _supabase_env(monkeypatch)
    calls = _capture_urlopen(monkeypatch)
    assert storage.archive_measurement("rid", **ARCHIVE_ARGS) is True
    assert [c["url"].rsplit("/", 1)[1] for c in calls] == ["front.bin", "side.bin", "envelope.json.bin"]
    assert {c["headers"]["Content-type"] for c in calls} == {"application/octet-stream"}
    sealed_overhead = 48  # X25519 ephemeral public key (32) + Poly1305 tag (16)
    assert len(calls[0]["body"]) == len(ARCHIVE_ARGS["front_bytes"]) + sealed_overhead
    assert calls[0]["body"] != ARCHIVE_ARGS["front_bytes"]


def test_supabase_http_error_returns_false(storage, monkeypatch):
    import urllib.error
    import urllib.request

    _supabase_env(monkeypatch, POINTSX_ARCHIVE_ENCRYPT="off")

    def failing(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    assert storage.archive_measurement("rid", **ARCHIVE_ARGS) is False


# ── S3-compatible (fake boto3) ────────────────────────────────────────────────

@pytest.fixture
def fake_boto3(monkeypatch):
    botocore_exc = pytest.importorskip("botocore.exceptions")
    made = {"clients": [], "puts": [], "fail_codes": []}

    class FakeClient:
        def put_object(self, **kw):
            if made["fail_codes"]:
                code = made["fail_codes"].pop(0)
                raise botocore_exc.ClientError({"Error": {"Code": code, "Message": "x"}}, "PutObject")
            made["puts"].append(kw)

    def client(service, **kw):
        made["clients"].append({"service": service, **kw})
        return FakeClient()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return made


def _s3_env(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("S3_BUCKET", "bkt")
    monkeypatch.setenv("S3_PREFIX", "pfx")


def test_s3_upload_keys_metadata_and_retry(storage, monkeypatch, fake_boto3):
    _s3_env(monkeypatch)
    fake_boto3["fail_codes"] = ["SignatureDoesNotMatch"]  # transient on the provider → retried
    assert storage.archive_measurement("rid", **ARCHIVE_ARGS) is True
    assert len(fake_boto3["clients"]) == 1 and fake_boto3["clients"][0]["region_name"] == "us-east-1"
    puts = fake_boto3["puts"]
    assert [p["Key"] for p in puts] == ["pfx/measurements/rid/front.jpg", "pfx/measurements/rid/side.png",
                                        "pfx/measurements/rid/envelope.json"]
    assert [p["ContentType"] for p in puts] == ["image/jpeg", "image/png", "application/json"]
    assert all(p["Bucket"] == "bkt" and p["ContentLength"] == len(p["Body"]) for p in puts)
    assert puts[0]["Metadata"] == ARCHIVE_ARGS["metadata"]

    monkeypatch.setenv("POINTSX_S3_NO_METADATA", "1")
    fake_boto3["puts"].clear()
    assert storage.archive_measurement("rid2", **ARCHIVE_ARGS) is True
    assert fake_boto3["puts"][0]["Metadata"] == {}


def test_s3_permanent_error_returns_false(storage, monkeypatch, fake_boto3):
    _s3_env(monkeypatch)
    fake_boto3["fail_codes"] = ["AccessDenied"]
    assert storage.archive_measurement("rid", **ARCHIVE_ARGS) is False
