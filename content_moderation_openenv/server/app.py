# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# FastAPI application for the Content Moderation Openenv Environment.

# This module creates an HTTP server that exposes the ContentModerationOpenenvEnvironment
# over HTTP and WebSocket endpoints, compatible with EnvClient.

# Endpoints:
#     - POST /reset: Reset the environment
#     - POST /step: Execute an action
#     - GET /state: Get current environment state
#     - GET /schema: Get action/observation schemas
#     - WS /ws: WebSocket endpoint for persistent sessions

# Usage:
#     # Development (with auto-reload):
#     uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

#     # Production:
#     uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

#     # Or run directly:
#     python -m server.app
# """

# try:
#     from openenv.core.env_server.http_server import create_app
# except Exception as e:  # pragma: no cover
#     raise ImportError(
#         "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
#     ) from e

# try:
#     from ..models import ContentModerationOpenenvAction, ContentModerationOpenenvObservation
#     from .content_moderation_openenv_environment import ContentModerationOpenenvEnvironment
# except ModuleNotFoundError:
#     from models import ContentModerationOpenenvAction, ContentModerationOpenenvObservation
#     from server.content_moderation_openenv_environment import ContentModerationOpenenvEnvironment


# # Create the app with web interface and README integration
# app = create_app(
#     ContentModerationOpenenvEnvironment,
#     ContentModerationOpenenvAction,
#     ContentModerationOpenenvObservation,
#     env_name="content_moderation_openenv",
#     max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
# )


# def main(host: str = "0.0.0.0", port: int = 8000):
#     """
#     Entry point for direct execution via uv run or python -m.

#     This function enables running the server without Docker:
#         uv run --project . server
#         uv run --project . server --port 8001
#         python -m content_moderation_openenv.server.app

#     Args:
#         host: Host address to bind to (default: "0.0.0.0")
#         port: Port number to listen on (default: 8000)

#     For production deployments, consider using uvicorn directly with
#     multiple workers:
#         uvicorn content_moderation_openenv.server.app:app --workers 4
#     """
#     import uvicorn

#     uvicorn.run(app, host=host, port=port)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()
#     main(port=args.port)

"""
FastAPI server for the Content Moderation OpenEnv environment.
Exposes the standard OpenEnv REST interface + a web UI for demonstration.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ..models import Action, ActionType, ContentLabel
    from .content_moderation_openenv_environment import ContentModerationOpenenvEnvironment
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import Action, ActionType, ContentLabel
    from server.content_moderation_openenv_environment import ContentModerationOpenenvEnvironment

# Alias for compatibility
ContentModerationEnv = ContentModerationOpenenvEnvironment

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Content Moderation OpenEnv",
    description="Real-world content moderation environment for AI agent training & evaluation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global environment instance (single session for demo; production = per-session)
_env = None
_current_obs = None


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "basic_triage"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: str
    label: str
    confidence: float = 0.8
    reasoning: str = ""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    with open("ui/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "env": "content-moderation-env", "version": "1.0.0"}


@app.get("/info")
async def info():
    """Return environment metadata from openenv.yaml."""
    import yaml
    with open("openenv.yaml", "r") as f:
        meta = yaml.safe_load(f)
    return meta


@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset the environment for a new episode."""
    global _env, _current_obs
    _env = ContentModerationEnv(task_id=req.task_id)
    try:
        obs = _env.reset()
        _current_obs = obs
        return {
            "observation": obs.model_dump(),
            "message": f"Episode started. Task: {req.task_id}. {obs.queue_size} posts in queue."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    """Apply an action to the current post."""
    global _current_obs
    try:
        action = Action(
            action=ActionType(req.action),
            label=ContentLabel(req.label),
            confidence=req.confidence,
            reasoning=req.reasoning,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid action or label: {e}")

    try:
        result = _env.step(action)
        _current_obs = result.observation
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state():
    """Return full serialisable episode state."""
    try:
        s = _env.state()
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "basic_triage",
                "name": "Basic Content Triage",
                "difficulty": "easy",
                "description": "Classify 10 posts into SAFE / VIOLATING / BORDERLINE categories.",
                "queue_size": 10,
                "focus": "Label accuracy"
            },
            {
                "id": "policy_enforcement",
                "name": "Policy Enforcement with Actions",
                "difficulty": "medium",
                "description": "Choose correct enforcement actions for 10 nuanced policy cases.",
                "queue_size": 10,
                "focus": "Action selection + label accuracy"
            },
            {
                "id": "adversarial_moderation",
                "name": "Adversarial & Evasion Detection",
                "difficulty": "hard",
                "description": "Detect obfuscation, dog-whistles, and coded language in 10 adversarial posts.",
                "queue_size": 10,
                "focus": "Evasion detection + nuanced classification"
            }
        ]
    }


@app.get("/action_space")
async def action_space():
    return {
        "action": [a.value for a in ActionType],
        "label": [l.value for l in ContentLabel],
        "confidence": {"type": "float", "min": 0.0, "max": 1.0},
        "reasoning": {"type": "string", "max_length": 500}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)