"""webui.bootstrap.create_app — the assembled app exposes exactly the expected routes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXPECTED_ROUTES = sorted([
    ("GET", "/"),
    ("GET", "/dataset.html"),
    ("GET", "/api/health"),
    ("GET", "/api/keepalive"),
    ("POST", "/api/measure"),
    ("POST", "/api/measure/mock"),
    ("POST", "/api/tts"),
])


def _api_routes(app):
    # Read the OpenAPI schema, not app.routes: FastAPI >= 0.141 keeps included routers as lazy
    # _IncludedRouter entries, so app.routes no longer lists the endpoints themselves.
    paths = app.openapi()["paths"]
    return sorted((method.upper(), path) for path, ops in paths.items() for method in ops)


def test_routes_are_exactly_the_public_api(monkeypatch):
    monkeypatch.delenv("POINTSX_VERCEL", raising=False)
    from webui.bootstrap import create_app

    app = create_app(use_lifespan=False)
    assert _api_routes(app) == EXPECTED_ROUTES
    assert app.router.lifespan_context is not None  # default no-op context, not the model loader
    assert app.title == "FitMeasure AI WebUI"


def test_entrypoint_uses_the_factory():
    import webui.app

    assert _api_routes(webui.app.app) == EXPECTED_ROUTES
