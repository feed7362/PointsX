"""Run the PointsX web UI: `python -m webui` or console script `pointsx-web`.

Server bind:
    POINTSX_WEB_HOST    default 127.0.0.1
    POINTSX_WEB_PORT    default 8000

Model loading (consumed by the FastAPI lifespan in webui.app):
    POINTSX_POSE_MODEL_CUSTOM path to 16-keypoint pose .pt (default: models/pose-cus.pt)
    POINTSX_POSE_MODEL_COCO   path to COCO-17 pose .pt (default: models/yolo26-pose.pt)
    POINTSX_POSE_MODEL        legacy: overrides POINTSX_POSE_MODEL_CUSTOM when set
    POINTSX_SEG_MODEL         path to segmentation .pt (default: models/yolo12l-person-seg-extended.pt)
    POINTSX_REGRESSION_MODEL  path to regression .pt
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

Defaults load custom 16-pt pose (models/pose-cus.pt), COCO pose (models/yolo26-pose.pt),
and segmentation (models/yolo12l-person-seg-extended.pt). The UI lets you pick which
pose backend to use per request.

Override via env vars when needed:

  POINTSX_POSE_MODEL_CUSTOM=models/pose-cus.pt \\
  POINTSX_POSE_MODEL_COCO=models/yolo26-pose.pt \\
  POINTSX_SEG_MODEL=models/yolo12l-person-seg-extended.pt \\
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
