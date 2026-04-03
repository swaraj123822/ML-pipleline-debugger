"""
client.py — MLDebuggerEnv WebSocket Client
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

import websockets
from openenv.core.env_client import EnvClient

from models import (
    MLDebuggerObservation,
    MLDebuggerState,
    StepResult,
)

_DEFAULT_BASE_URL = os.environ.get(
    "MLDBG_BASE_URL",
    os.environ.get("HF_SPACE_URL", "http://localhost:7860"),
)

def _http_to_ws(url: str) -> str:
    url = url.rstrip("/")
    if url.startswith("https://"):
        return url.replace("https://", "wss://", 1) + "/ws"
    if url.startswith("http://"):
        return url.replace("http://", "ws://", 1) + "/ws"
    if not url.endswith("/ws"):
        return url + "/ws"
    return url


class MLDebuggerEnv(EnvClient):

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._ws_url = _http_to_ws(base_url)
        self._ws = None

    async def __aenter__(self) -> "MLDebuggerEnv":
        self._ws = await websockets.connect(self._ws_url)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    def _step_payload(self, action: Any) -> dict:
        if hasattr(action, "model_dump"):
            return action.model_dump(mode="json")
        return dict(action)

    def _parse_result(self, data: dict) -> StepResult:
        return StepResult(**data)

    def _parse_state(self, data: dict) -> MLDebuggerState:
        return MLDebuggerState(**data)

    async def reset(self, task_id: Literal["easy", "medium", "hard"] = "easy", **kwargs: Any) -> MLDebuggerObservation:
        response = await self._send({"method": "reset", "task_id": task_id})
        return MLDebuggerObservation(**response)

    async def step(self, action: Any) -> StepResult:
        action_dict = self._step_payload(action)
        response = await self._send({"method": "step", "action": action_dict})
        return self._parse_result(response)

    async def state(self) -> MLDebuggerState:
        response = await self._send({"method": "state"})
        return self._parse_state(response)

    def sync(self) -> "SyncMLDebuggerEnv":
        return SyncMLDebuggerEnv(self)

    async def _send(self, message: dict) -> dict:
        if self._ws is None:
            raise RuntimeError("Not connected. Use 'async with MLDebuggerEnv(...) as env:'")
        await self._ws.send(json.dumps(message))
        raw = await self._ws.recv()
        data = json.loads(raw)

        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        if "result" not in data:
            raise RuntimeError(f"Unexpected server response: {data}")

        return data["result"]


class SyncMLDebuggerEnv:

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