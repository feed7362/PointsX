import sys
import os
from pathlib import Path

# Resolve the absolute path to the 'src' directory
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Signal to the app that we are running inside Vercel's serverless environment.
# This disables the StaticFiles mount (Vercel serves /static directly via rewrites).
os.environ.setdefault("POINTSX_VERCEL", "1")

# Import the FastAPI application instance for Vercel's ASGI builder
from webui.app import app  # noqa: E402, F401
