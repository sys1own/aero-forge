"""Thin async client for the batch analytics HTTP service."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import aiohttp


class BatchAnalyticsClient:
    """Client for submitting record batches to a running service."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")

    async def aggregate(
        self,
        records: List[Mapping[str, Any]],
        window: int,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a batch to `/aggregate` and return the JSON response."""
        payload = {
            "records": records,
            "window": window,
            "threshold": threshold,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/aggregate", json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def health(self) -> Dict[str, Any]:
        """Hit the `/health` endpoint."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()
