#!/usr/bin/env python3
"""Convenience entrypoint to start the batch analytics HTTP service."""

from __future__ import annotations

from aiohttp import web

from batch_analytics.service import app


if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
