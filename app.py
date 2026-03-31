#!/usr/bin/env python3
"""
FastAPI server for the AI Customer Support Ticket System.

Exposes the OpenEnv interface over HTTP for Hugging Face Spaces
deployment and programmatic access.

Endpoints:
    GET  /reset?task_id=<id>  — Reset environment, return initial observation
    POST /step                — Submit action, return (observation, reward, done, info)
    GET  /state               — Current environment state
    GET  /tasks               — List available tasks
    GET  /health              — Health check
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import CustomerSupportEnv
from env.models import Action, ActionType


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support — OpenEnv",
    description=(
        "A realistic customer support simulation environment for AI agents. "
        "Submit actions and receive evaluations on correctness, politeness, "
        "and completeness."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session)
env = CustomerSupportEnv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    """Request body for POST /step."""
    action_type: str = "respond"
    response: str = ""
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "respond",
                "response": "Thank you for reaching out. I'd be happy to help you with your order status.",
                "metadata": {},
            }
        }


class StepResponse(BaseModel):
    """Response body for POST /step."""
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AI Customer Support — OpenEnv",
        "version": "1.0.0",
        "endpoints": [
            "GET  /reset?task_id=<id>",
            "POST /step",
            "GET  /state",
            "GET  /tasks",
            "GET  /health",
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {"tasks": CustomerSupportEnv.available_tasks()}


@app.get("/reset")
async def reset(task_id: str = Query(default="easy_order_status")):
    """
    Reset the environment with the specified task.

    Args:
        task_id: One of easy_order_status, medium_angry_refund, hard_multi_issue.
    """
    try:
        obs = env.reset(task_id=task_id)
        return JSONResponse(content=obs.model_dump(mode="json"))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Submit an action and receive the next observation, reward, and status.

    The request body should contain:
        action_type: One of respond, escalate, resolve, request_info, refund
        response: The text response to send to the customer
        metadata: Optional extra data
    """
    if env.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
        )

    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{request.action_type}'. "
                   f"Valid types: {[a.value for a in ActionType]}",
        )

    action = Action(
        action_type=action_type,
        response=request.response,
        metadata=request.metadata,
    )

    obs, reward, done, info = env.step(action)

    return StepResponse(
        observation=obs.model_dump(mode="json"),
        reward=reward.model_dump(mode="json"),
        done=done,
        info=info,
    )


@app.get("/state")
async def state():
    """Get the current environment state."""
    return JSONResponse(content=env.state())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
