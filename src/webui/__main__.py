"""Run the PointsX web UI: `python -m webui` or console script `pointsx-web`.

Server bind:
    POINTSX_WEB_HOST    default 127.0.0.1
    POINTSX_WEB_PORT    default 8000

Model loading (consumed by the FastAPI lifespan in webui.app):
    POINTSX_POSE_MODEL        path to YOLO11n-pose .pt
                              default: runs/pose/best.pt (your fine-tuned weights)
    POINTSX_SEG_MODEL         path to YOLO11n-seg .pt
                              default: runs/seg/best.pt
    POINTSX_REGRESSION_MODEL  path to circumference_regressor.pt
                              default: models/circumference_regressor.pt if present;
                              set to "" to force the Ramanujan ellipse fallback.
    POINTSX_DEVICE            "auto" | "cpu" | "cuda" | "0" | …
                              default: auto

If a model fails to load, the server still starts; `/api/measure` returns 503
until the env vars are corrected and the server is restarted.
"""

from __future__ import annotations

import argparse
import os


_DESCRIPTION = """\
PointsX measurement web UI (FastAPI + static assets).

Defaults assume your trained weights are at runs/pose/best.pt, runs/seg/best.pt,
and models/circumference_regressor.pt — just run `pointsx-web --reload`.

Override via env vars when needed:

  POINTSX_POSE_MODEL=runs/pose/best.pt \\
  POINTSX_SEG_MODEL=runs/seg/best.pt \\
  POINTSX_REGRESSION_MODEL=models/circumference_regressor.pt \\
  POINTSX_DEVICE=cpu \\
  pointsx-web --reload
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("POINTSX_WEB_HOST", "127.0.0.1"),
        help="Bind host (env: POINTSX_WEB_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("POINTSX_WEB_PORT", "8000")),
        help="Bind port (env: POINTSX_WEB_PORT)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Dev auto-reload (do not use in production)",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "webui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
