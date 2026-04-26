"""Run the PointsX web UI: `python -m webui` or console script `pointsx-web`."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="PointsX mock measurement web UI (FastAPI + static assets).")
    parser.add_argument("--host", default=os.environ.get("POINTSX_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POINTSX_WEB_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (do not use in production)")
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
