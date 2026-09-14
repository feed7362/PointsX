"""App assembly (the equivalent of Video_streaming's ``app/``): factory, lifespan, routers, middleware, errors."""
from webui.bootstrap.factory import create_app

__all__ = ["create_app"]
