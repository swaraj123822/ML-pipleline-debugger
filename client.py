"""
client.py — MLDebuggerEnv WebSocket Client

What training code and the baseline script import.
Wraps the WebSocket connection — caller sees clean Python methods.

Usage (async):
    async with MLDebuggerEnv(base_url="http://localhost:7860") as env:
        obs = await env.reset(task_id="easy")
        result = await env.step({"action_type": "fix_reshape", "layer": "flatten", "new_shape": [2304]})
        state = await env.state()

Usage (sync — for scripts and notebooks):
    with MLDebuggerEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task_id="easy")
        result = env.step(action)

The base_url defaults to the Hugging Face Space URL once deployed.
Override with the HF_SPACE_URL or MLDBG_BASE_URL environment variables,
or pass base_url= explicitly.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

import websockets
from openenv.core.env_client import EnvClient,  SyncEnvClient

from models import (
    MLDebuggerObservation,
    MLDebuggerState,
    StepResult,
)

# Default base URL — overridden by env var or constructor argument
_DEFAULT_BASE_URL = os.environ.get(
    "MLDBG_BASE_URL",
    os.environ.get("HF_SPACE_URL", "http://localhost:7860"),
)


def _http_to_ws(url: str) -> str:
    """Convert http(s):// base URL to ws(s):// WebSocket URL."""
    url = url.rstrip("/")
    if url.startswith("https://"):
        return url.replace("https://", "wss://", 1) + "/ws"
    if url.startswith("http://"):
        return url.replace("http://", "ws://", 1) + "/ws"
    # Already ws/wss
    if not url.endswith("/ws"):
        return url + "/ws"
    return url


class MLDebuggerEnv(EnvClient):
    """
    Async WebSocket client for the ML Pipeline Debugger environment.

    Implements the OpenEnv EnvClient interface:
        reset(**kwargs) -> MLDebuggerObservation
        step(action)    -> StepResult
        state()         -> MLDebuggerState
        sync()          -> SyncMLDebuggerEnv  (sync wrapper)

    Context manager usage:
        async with MLDebuggerEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset(task_id="easy")
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._ws_url = _http_to_ws(base_url)
        self._ws = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MLDebuggerEnv":
        self._ws = await websockets.connect(self._ws_url)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ------------------------------------------------------------------
    # OpenEnv async interface
    # ------------------------------------------------------------------

    async def reset(self, task_id: Literal["easy", "medium", "hard"] = "easy", **kwargs: Any) -> MLDebuggerObservation:
        """Reset the environment and return the initial observation."""
        response = await self._send({"method": "reset", "task_id": task_id})
        return MLDebuggerObservation(**response)

    async def step(self, action: Any) -> StepResult:
        """
        Submit an action. Returns a StepResult.

        action can be:
          - A Pydantic model (TuneHyperparameters, FixReshape, etc.)
          - A plain dict with action_type key
        """
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump(mode="json")
        else:
            action_dict = dict(action)

        response = await self._send({"method": "step", "action": action_dict})
        return StepResult(**response)

    async def state(self) -> MLDebuggerState:
        """Return the current internal episode state."""
        response = await self._send({"method": "state"})
        return MLDebuggerState(**response)

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "SyncMLDebuggerEnv":
        """Return a synchronous wrapper for use in scripts and notebooks."""
        return SyncMLDebuggerEnv(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send(self, message: dict) -> dict:
        """Send a message and wait for the server's response."""
        if self._ws is None:
            raise RuntimeError(
                "Not connected. Use 'async with MLDebuggerEnv(...) as env:' "
                "or call __aenter__() manually."
            )
        await self._ws.send(json.dumps(message))
        raw = await self._ws.recv()
        data = json.loads(raw)

        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        if "result" not in data:
            raise RuntimeError(f"Unexpected server response: {data}")

        return data["result"]


class SyncMLDebuggerEnv(SyncEnvClient):
    """
    Synchronous wrapper around MLDebuggerEnv.
    Use this in scripts, notebooks, or the baseline agent.

    with MLDebuggerEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task_id="medium")
        result = env.step(action)
    """

    def __init__(self, async_client: MLDebuggerEnv) -> None:
        self._async = async_client
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> "SyncMLDebuggerEnv":
        self._loop.run_until_complete(self._async.__aenter__())
        return self

    def __exit__(self, *args: Any) -> None:
        self._loop.run_until_complete(self._async.__aexit__(*args))
        self._loop.close()

    def reset(self, task_id: Literal["easy", "medium", "hard"] = "easy", **kwargs: Any) -> MLDebuggerObservation:
        return self._loop.run_until_complete(self._async.reset(task_id=task_id, **kwargs))

    def step(self, action: Any) -> StepResult:
        return self._loop.run_until_complete(self._async.step(action))

    def state(self) -> MLDebuggerState:
        return self._loop.run_until_complete(self._async.state())