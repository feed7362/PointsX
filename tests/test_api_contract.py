"""HTTP contract of the web app, pinned BEFORE the webui decomposition.

Every test talks to the app only through `webui.app:app` (the entrypoint both
deploys use) and `app.state.pipeline`, so the tests stay valid while the
internals move into bootstrap/api/services/schemas. Status codes, response
shapes and the Ukrainian error texts are part of the contract: the frontend
shows `detail` to users.

No model weights and no photos: a fake pipeline returns results built from the
procedural bodies in tests/fixtures/synthetic_bodies.py.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

ENV_KEYS = (
    "POINTSX_VERCEL", "POINTSX_INFERENCE_ENDPOINT", "POINTSX_DATASET_DIR", "POINTSX_TTS_DISABLE",
    "CRON_SECRET", "CORS_ALLOW_ORIGINS", "LOCAL_DATA_DIR",
)


def _generator():
    if "synthetic_bodies" in sys.modules:
        return sys.modules["synthetic_bodies"]
    spec = importlib.util.spec_from_file_location("synthetic_bodies", FIXTURES / "synthetic_bodies.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fresh_app(monkeypatch, **env):
    """Import webui.app from scratch under `env` (module-level config is read at import)."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in [m for m in sys.modules if m == "webui" or m.startswith("webui.")]:
        del sys.modules[name]
    return importlib.import_module("webui.app").app


def _png(width: int = 64, height: int = 96) -> bytes:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _form(height="175", sex="female", **extra):
    data = {"height_cm": height, "sex": sex}
    data.update(extra)
    return data


def _files(front: bytes | None = None, side: bytes | None = None, ctype="image/png"):
    front = _png() if front is None else front
    side = _png() if side is None else side
    return {"front": ("front.png", front, ctype), "side": ("side.png", side, ctype)}


def _inference_result(body_index: int = 1):
    """InferenceResult for a synthetic body, computed exactly like the production pipeline."""
    gen = _generator()
    from pointsx.calibration import calibrate
    from pointsx.circumference import estimate_circumferences
    from pointsx.measurements import extract_measurements
    from pointsx.postprocess import validate_measurements
    from webui.infrastructure.inference import InferenceResult

    b = gen.BODIES[body_index]
    fkp, fmask = gen.front_view(b)
    skp, smask = gen.side_view(b)
    cal = calibrate(fkp, skp, b.height_cm)
    bm = validate_measurements(estimate_circumferences(extract_measurements(fkp, skp, fmask, smask, cal), None))
    return b, InferenceResult(body=bm, front_kp=fkp, side_kp=skp, front_mask=fmask, side_mask=smask,
                              cal=cal, has_regressor=False, pose_backend="coco")


class _Models:
    def __init__(self, backends):
        self._backends = set(backends)

    def available_pose_backends(self):
        return set(self._backends)


class FakePipeline:
    """Stands in for WebuiPipeline: records calls, returns a synthetic result or raises."""

    def __init__(self, backends=("coco",), result=None, error: Exception | None = None, preview=None):
        self.models = _Models(backends)
        self.result, self.error, self.preview_result = result, error, preview
        self.calls: list[dict] = []

    def measure(self, front_img, side_img, height_cm, *, pose_backend="coco", timings=None):
        self.calls.append({"front_shape": front_img.shape, "side_shape": side_img.shape,
                           "height_cm": height_cm, "pose_backend": pose_backend})
        if self.error is not None:
            raise self.error
        return self.result

    def preview(self, front_img, side_img, *, pose_backend="coco"):
        return self.preview_result


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    app = _fresh_app(monkeypatch, POINTSX_DATASET_DIR=str(tmp_path / "dataset"))
    app.state.pipeline = None  # lifespan (model loading) is not run in tests
    app.state.pipeline_load_error = None
    return TestClient(app)


# ── pages, health, headers ────────────────────────────────────────────────────

def test_index_and_dataset_pages(client):
    for path in ("/", "/dataset.html"):
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path


def test_health_shape_without_pipeline(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"service": "pointx-backend", "status": "ok", "proxy_mode": False,
                        "pipeline_ready": False, "pose_backends": [], "pipeline_load_error": None}


def test_health_reports_loaded_backends(client):
    client.app.state.pipeline = FakePipeline(backends=("coco", "custom"))
    body = client.get("/api/health").json()
    assert body["pipeline_ready"] is True and body["pose_backends"] == ["coco", "custom"]


def test_camera_permissions_and_cors_headers(client):
    r = client.get("/api/health")
    assert r.headers.get("permissions-policy") == "camera=(self)"
    pre = client.options("/api/measure", headers={"Origin": "https://example.com",
                                                   "Access-Control-Request-Method": "POST"})
    assert pre.headers.get("access-control-allow-origin") == "*"


def test_keepalive_outside_proxy_mode(client, monkeypatch):
    assert client.get("/api/keepalive").json() == {"target": "self", "ok": True}
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    assert client.get("/api/keepalive").status_code == 401
    assert client.get("/api/keepalive", headers={"Authorization": "Bearer s3cret"}).status_code == 200


# ── mock ───────────────────────────────────────────────────────────────────────

def test_mock_envelope(client):
    r = client.post("/api/measure/mock", data=_form(height="170", sex="male"))
    assert r.status_code == 200
    env = r.json()
    assert env["schema"] == "pointsx.measurement.envelope" and env["schema_version"] == 2
    assert env["pipeline"]["source"] == "mock" and env["subject"] == {
        "height_cm": 170.0, "sex": "male", "age_band": "adult", "posture_flags": []}
    assert len(env["measurements"]) == 18
    assert set(env["measurements"][0]) == {"id", "label_uk", "value_cm", "uncertainty_cm", "confidence",
                                           "source", "quality_flags"}
    assert env["warnings"] == ["Тестовий режим: зображення й моделі не використовувалися."]


# ── /api/measure: validation and errors ───────────────────────────────────────

def test_measure_form_validation_is_ukrainian(client):
    r = client.post("/api/measure", data={}, files=_files())
    assert r.status_code == 422
    assert "Зріст (см): значення не передано." in r.json()["detail"]
    assert "Стать: значення не передано." in r.json()["detail"]

    low = client.post("/api/measure", data=_form(height="90"), files=_files())
    assert low.status_code == 422 and "Зріст (см): занадто мале значення (мінімум 100.0)." in low.json()["detail"]
    bad_sex = client.post("/api/measure", data=_form(sex="robot"), files=_files())
    assert bad_sex.status_code == 422 and "Стать: недопустиме значення." in bad_sex.json()["detail"]


def test_measure_without_pipeline_is_503(client):
    r = client.post("/api/measure", data=_form(), files=_files())
    assert r.status_code == 503
    assert r.json()["detail"].startswith("Неможливо виконати замір: моделі не завантажені на сервері.")


def test_measure_unavailable_pose_backend_is_503(client):
    client.app.state.pipeline = FakePipeline(backends=("coco",))
    r = client.post("/api/measure", data=_form(pose_backend="custom"), files=_files())
    assert r.status_code == 503 and "pose-cus.pt (16 точок)" in r.json()["detail"]


@pytest.mark.parametrize(("front", "ctype", "expected"), [
    (b"", "image/png", "Анфас: файл порожній."),
    (b"hello world, not an image", "text/plain", "Анфас: недопустимий тип вмісту"),
    (b"hello world, not an image", "image/png", "Анфас: очікується JPEG, PNG або WebP."),
    (b"\x89PNG\r\n\x1a\n" + b"0" * (5 * 1024 * 1024), "image/png", "Анфас: файл завеликий (ліміт 5 МБ)."),
], ids=["empty", "text-content-type", "not-an-image", "too-large"])
def test_measure_upload_validation(client, front, ctype, expected):
    client.app.state.pipeline = FakePipeline()
    files = {"front": ("front.bin", front, ctype), "side": ("side.png", _png(), "image/png")}
    r = client.post("/api/measure", data=_form(), files=files)
    assert r.status_code == 400 and r.json()["detail"].startswith(expected), r.json()


def test_measure_pipeline_value_error_is_translated(client):
    client.app.state.pipeline = FakePipeline(error=ValueError("No person detected in front image"))
    r = client.post("/api/measure", data=_form(), files=_files())
    assert r.status_code == 400
    assert r.json()["detail"].startswith("На знімку анфасу не виявлено людину.")


def test_measure_unexpected_error_is_500(client):
    client.app.state.pipeline = FakePipeline(error=RuntimeError("boom"))
    r = client.post("/api/measure", data=_form(), files=_files())
    assert r.status_code == 500 and r.json()["detail"] == "Помилка під час обчислення мірок: boom"


# ── /api/measure: success and calibration-failure paths ───────────────────────

def test_measure_success_envelope_matches_geometry_snapshot(client, tmp_path):
    body, result = _inference_result(body_index=1)
    fake = FakePipeline(result=result)
    client.app.state.pipeline = fake
    r = client.post("/api/measure?with_viz=0", data=_form(height=str(body.height_cm), sex=body.sex),
                    files=_files())
    assert r.status_code == 200, r.text
    env = r.json()

    expected = json.loads((FIXTURES / "expected" / "synthetic_snapshot.json").read_text(encoding="utf-8"))
    assert {m["id"]: m["value_cm"] for m in env["measurements"]} == expected[body.body_id]["envelope"]
    assert env["derived"] == {}  # with_viz=0 skips the overlay render
    assert env["pipeline"]["pose_backend"] == "coco" and env["subject"]["sex"] == body.sex
    assert fake.calls == [{"front_shape": (96, 64, 3), "side_shape": (96, 64, 3),
                           "height_cm": body.height_cm, "pose_backend": "coco"}]

    saved = sorted(p.name[0] + p.suffix for p in (tmp_path / "dataset").iterdir())
    assert saved == ["a.png", "p.png"]  # one capture pair, original bytes and extension


def test_measure_downscales_large_uploads_before_the_pipeline(client):
    _, result = _inference_result()
    fake = FakePipeline(result=result)
    client.app.state.pipeline = fake
    big = _png(width=2000, height=3000)
    r = client.post("/api/measure?with_viz=0", data=_form(), files=_files(front=big, side=big))
    assert r.status_code == 200, r.text
    assert fake.calls[0]["front_shape"] == (1280, 853, 3)


def test_measure_calibration_failure_returns_visualization_only_envelope(client):
    _, result = _inference_result()
    msg = "Cannot calibrate front view: insufficient visible keypoints"
    client.app.state.pipeline = FakePipeline(error=ValueError(msg), preview=result)
    r = client.post("/api/measure", data=_form(), files=_files())
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["measurements"] == [] and env["pipeline"]["model_version"] == "visualization-only"
    assert env["warnings"][0].startswith("Недостатньо видимих ключових точок на анфасі")


def test_measure_dataset_save_failure_is_a_warning_not_an_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file where the dataset directory should be", encoding="utf-8")
    app = _fresh_app(monkeypatch, POINTSX_DATASET_DIR=str(blocker))
    _, result = _inference_result()
    app.state.pipeline, app.state.pipeline_load_error = FakePipeline(result=result), None
    r = TestClient(app).post("/api/measure?with_viz=0", data=_form(), files=_files())
    assert r.status_code == 200, r.text
    assert r.json()["warnings"][-1].startswith("Не вдалося зберегти знімки у папку датасету")


# ── TTS ────────────────────────────────────────────────────────────────────────

def test_tts_disabled_and_validation(client, monkeypatch):
    monkeypatch.setenv("POINTSX_TTS_DISABLE", "1")
    r = client.post("/api/tts", json={"text": "Привіт"})
    assert r.status_code == 503 and r.json()["detail"] == "Синтез мовлення вимкнено на сервері."
    assert client.post("/api/tts", json={"text": ""}).status_code == 422


# ── Vercel proxy mode ─────────────────────────────────────────────────────────

def test_proxy_mode_contract(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    app = _fresh_app(monkeypatch, POINTSX_VERCEL="1", POINTSX_INFERENCE_ENDPOINT="http://127.0.0.1:9",
                     POINTSX_DATASET_DIR=str(tmp_path / "dataset"))
    with TestClient(app) as c:  # lifespan runs: in proxy mode it must not load anything
        assert c.get("/api/health").json()["proxy_mode"] is True
        r = c.post("/api/measure", data=_form(), files=_files())
        assert r.status_code == 502
        assert r.json()["detail"].startswith("Не вдалося зʼєднатися з сервером інференсу")
        assert c.get("/api/keepalive").status_code == 502
        assert c.post("/api/measure/mock", data=_form()).status_code == 200
    assert not (tmp_path / "dataset").exists()  # proxy mode never stores photos locally
