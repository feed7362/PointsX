"""HTTP routers. Thin: parse the request, call a service, return a schema."""
from webui.api import health, measure, pages, tts

# Registration order is part of the app's behaviour (route matching); keep it stable.
ROUTERS = (pages.router, health.router, measure.router, tts.router)

__all__ = ["ROUTERS"]
