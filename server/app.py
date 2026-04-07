"""
server/app.py — FastAPI + WebSocket Server

Exposes the MLDebuggerEnvironment over WebSocket at /ws.
Each connection gets its own isolated environment instance.

Endpoints:
  GET  /          → welcome message
  GET  /health    → {"status": "ok"}
  GET  /info      → environment metadata (tasks, action space, etc.)
  WS   /ws        → persistent session: reset / step / state messages

WebSocket message protocol (JSON):
  Client → Server:
    {"method": "reset", "task_id": "easy"}
    {"method": "step",  "action": {"action_type": "fix_reshape", ...}}
    {"method": "state"}

  Server → Client:
    {"result": <observation | step_result | state>}
    {"error": "<message>"}
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models import (
    AdjustLossWeights,
    AddAugmentation,
    FixReshape,
    TuneHyperparameters,
    ChangeOptimizer,
    ToggleLayerFreeze
)
from server.environment import MLDebuggerEnvironment
from server.tasks import TASK_REGISTRY

app = FastAPI(
    title="ML Pipeline Debugger",
    description=(
        "OpenEnv-compliant RL environment where an LLM agent debugs "
        "broken ML training pipelines across three difficulty levels."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "ML Pipeline Debugger",
        "version": "0.1.0",
        "description": "OpenEnv environment for LLM-driven ML pipeline debugging.",
        "websocket": "/ws",
        "health": "/health",
        "info": "/info",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
async def info() -> dict[str, Any]:
    return {
        "name": "ML Pipeline Debugger",
        "version": "0.1.0",
        "tasks": list(TASK_REGISTRY.values()),
        "action_space": {
            "type": "discriminated_union",
            "discriminator": "action_type",
            "variants": [
                {
                    "action_type": "tune_hyperparameters",
                    "fields": {"lr": "float (0,1)", "batch_size": "int power-of-2", "epochs": "int [1,50]"},
                },
                {
                    "action_type": "fix_reshape",
                    "fields": {"layer": "str", "new_shape": "list[int] max 4 dims"},
                },
                {
                    "action_type": "add_augmentation",
                    "fields": {"strategy": "dropout|weight_decay|truncate_sequence|horizontal_flip|mixup"},
                },
                {
                    "action_type": "adjust_loss_weights",
                    "fields": {"dice_weight": "float [0,1]", "ce_weight": "float [0,1] (must sum to 1.0)"},
                },
                {
                    "action_type": "change_optimizer",
                    "fields": {"optimizer": "Adam|SGD|RMSprop", "weight_decay": "float >= 0.0"},
                },
                {
                    "action_type": "toggle_layer_freeze",
                    "fields": {"layer_name": "str", "freeze": "bool"},
                },
            ],
        },
        "observation_space": {
            "task_id": "easy|medium|hard",
            "architecture_summary": "str",
            "tensor_shapes": "dict[str, list[int]]",
            "error_trace": "str|null",
            "metrics_history": "list[EpochMetrics]",
            "step_number": "int",
            "max_steps": "int (15)",
        },
        "reward": {
            "type": "dense",
            "range": [-1.0, 2.0],
            "schedule": {
                "+0.1": "per successful simulated epoch",
                "+1.0": "per 0.05 drop in validation loss",
                "-0.5": "crash / OOM / NaN loss",
                "-0.3": "repeated failed action (loop detection)",
                "+2.0": "task solved bonus",
            },
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Persistent WebSocket session. One env instance per connection.

    Supported methods:
      reset  → {"method": "reset", "task_id": "easy|medium|hard"}
      step   → {"method": "step", "action": {...}}
      state  → {"method": "state"}
    """
    await websocket.accept()
    env = MLDebuggerEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON.")
                continue

            method = message.get("method")

            if method == "reset":
                task_id = message.get("task_id", "easy")
                try:
                    obs = env.reset(task_id=task_id)
                    await _send_result(websocket, obs.model_dump(mode="json"))
                except Exception as e:
                    await _send_error(websocket, str(e))

            elif method == "step":
                raw_action = message.get("action")
                if raw_action is None:
                    await _send_error(websocket, "Missing 'action' field in step message.")
                    continue
                try:
                    result = env.step(raw_action)
                    await _send_result(websocket, result.model_dump(mode="json"))
                except Exception as e:
                    await _send_error(websocket, str(e))

            elif method == "state":
                try:
                    state = env.state
                    await _send_result(websocket, state.model_dump(mode="json"))
                except Exception as e:
                    await _send_error(websocket, str(e))

            else:
                await _send_error(
                    websocket,
                    f"Unknown method '{method}'. Valid methods: reset, step, state.",
                )

    except WebSocketDisconnect:
        pass  # Client disconnected cleanly — nothing to do


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _send_result(websocket: WebSocket, data: Any) -> None:
    await websocket.send_text(json.dumps({"result": data}))


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_text(json.dumps({"error": message}))


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()