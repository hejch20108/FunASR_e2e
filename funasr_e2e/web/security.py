from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost")) -> None:
        super().__init__(app)
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)

    async def dispatch(self, request: Request, call_next) -> Response:
        host = request.url.hostname
        if host is None or host.lower() not in self.allowed_hosts:
            return self._rejected(421, "HOST_REJECTED", "仅允许本机访问")
        if request.method in _UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and not self._is_allowed_origin(origin):
                return self._rejected(403, "ORIGIN_REJECTED", "请求来源不被允许")
        return await call_next(request)

    def _is_allowed_origin(self, origin: str) -> bool:
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and parsed.hostname is not None and parsed.hostname.lower() in self.allowed_hosts

    @staticmethod
    def _rejected(status_code: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})
