"""Async HTTP service exposing the batch analytics Rust core."""

from __future__ import annotations

from aiohttp import web

from batch_analytics import aggregate_batch, detect_outliers
from batch_analytics.utils import parse_records, serialize_aggregate


async def aggregate(request: web.Request) -> web.Response:
    """POST /aggregate — compute windowed aggregates and optional outliers."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")

        records = parse_records(body.get("records"))
        window = int(body["window"])
        threshold = float(body.get("threshold", 0.0))
    except (KeyError, ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

    results = aggregate_batch(records, window)

    if threshold > 0.0:
        outlier_indices = detect_outliers(results, threshold)
        for idx in outlier_indices:
            if idx < len(results):
                results[idx].outliers = [idx]

    payload = {
        "results": [serialize_aggregate(r) for r in results],
        "count": len(results),
    }
    return web.json_response(payload)


async def health(request: web.Request) -> web.Response:
    """GET /health — liveness probe."""
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    """Factory for the aiohttp application."""
    app = web.Application()
    app.router.add_post("/aggregate", aggregate)
    app.router.add_get("/health", health)
    return app


app = create_app()
