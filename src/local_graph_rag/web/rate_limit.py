"""Sliding-window in-memory rate limiter."""

import asyncio
import time
from collections import defaultdict

from local_graph_rag.settings import RATE_MAX_LOGIN_REQUESTS, RATE_MAX_REQUESTS, RATE_WINDOW_SECONDS

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_login_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_lock = asyncio.Lock()
_login_rate_lock = asyncio.Lock()


async def check_rate_limit(
    ip: str,
    *,
    buckets: dict[str, list[float]] = _rate_buckets,
    lock: asyncio.Lock = _rate_lock,
    max_requests: int = RATE_MAX_REQUESTS,
) -> bool:
    """Return True if the request is within rate limits, False if it should be rejected."""
    async with lock:
        now = time.monotonic()
        active = [t for t in buckets.get(ip, []) if now - t < RATE_WINDOW_SECONDS]
        if len(active) >= max_requests:
            return False
        active.append(now)
        buckets[ip] = active
        return True


async def check_login_rate_limit(ip: str) -> bool:
    """Return True if the login attempt is within the login rate limit."""
    return await check_rate_limit(
        ip,
        buckets=_login_rate_buckets,
        lock=_login_rate_lock,
        max_requests=RATE_MAX_LOGIN_REQUESTS,
    )


async def _sweep_expired(buckets: dict[str, list[float]], lock: asyncio.Lock) -> None:
    """Evict IPs whose entire window has expired so the dicts don't grow unboundedly."""
    while True:
        await asyncio.sleep(RATE_WINDOW_SECONDS)
        async with lock:
            now = time.monotonic()
            dead = [
                ip
                for ip, ts in buckets.items()
                if not any(now - t < RATE_WINDOW_SECONDS for t in ts)
            ]
            for ip in dead:
                del buckets[ip]


async def start_sweep_tasks() -> list[asyncio.Task[None]]:
    return [
        asyncio.create_task(_sweep_expired(_rate_buckets, _rate_lock)),
        asyncio.create_task(_sweep_expired(_login_rate_buckets, _login_rate_lock)),
    ]
