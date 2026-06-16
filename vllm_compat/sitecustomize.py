"""Compatibility patches loaded only by the vLLM subprocess.

vLLM 0.18.1 with FastAPI 0.137 / Starlette 1.3 can expose an included router
object that matches a request but has no ``path`` attribute. The bundled
``prometheus-fastapi-instrumentator`` assumes every matched route has ``path``,
which turns every HTTP route into a 500. Patch its route-name helper to skip
that assumption while leaving metrics middleware enabled.
"""


def _patch_prometheus_route_names() -> None:
    try:
        from prometheus_fastapi_instrumentator import routing
        from starlette.routing import Match, Mount
    except Exception:
        return

    def _safe_get_route_name(scope, routes, route_name=None):
        for route in routes:
            try:
                match, child_scope = route.matches(scope)
            except Exception:
                continue

            path = getattr(route, "path", route_name)
            if match == Match.FULL:
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and getattr(route, "routes", None):
                    child_route_name = _safe_get_route_name(
                        child_scope, route.routes, path
                    )
                    if child_route_name is None:
                        return None
                    return (path or "") + child_route_name
                return path

            if match == Match.PARTIAL and route_name is None:
                route_name = path
        return None

    routing._get_route_name = _safe_get_route_name


_patch_prometheus_route_names()
