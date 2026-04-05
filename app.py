"""
FastAPI server for SQL Query Generation OpenEnv.

Exposes the environment over HTTP for Hugging Face Spaces deployment
and programmatic access.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SQLAction, SQLObservation, Reward, SQLState
from pydantic import BaseModel
import random

class StepRequest(BaseModel):
    generated_sql: str
from sql_query_environment import SQLQueryEnv


# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="SQL Query Generation — OpenEnv",
    description=(
        "A real-world RL environment where agents learn to generate SQL queries "
        "from natural language. Submit queries and receive deterministic rewards."
    ),
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = SQLQueryEnv(db_path=":memory:")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SQL Query Generation — OpenEnv",
        "version": "1.0.0",
        "description": "Generate SQL queries from natural language descriptions",
        "endpoints": [
            "GET  /reset?question_id=<id>",
            "POST /step",
            "GET  /state",
            "GET  /health",
            "GET  /docs",
        ],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "sql-query-gen"}


@app.get("/reset")
async def reset(question_id: str = "q1"):
    """Reset environment with a new question.
    
    Args:
        question_id: One of q1-q5 (order status, total sales, etc.)
    
    Returns:
        Initial observation with question and schema
    """
    try:
        obs = env.reset(question_id=question_id)
        return JSONResponse(content=obs.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset_openenv():
    """OpenEnv POST /reset - random question."""
    q_ids = ["q1", "q2", "q3", "q4", "q5"]
    q_id = random.choice(q_ids)
    obs = env.reset(question_id=q_id)
    return obs.model_dump()


@app.post("/step")
async def step(action: SQLAction):
    """Submit a SQL query and receive reward.
    
    Args:
        action: SQLAction with query and optional reasoning
    
    Returns:
        Observation, reward, done, and info dict
    """
    try:
        obs, reward, done, info = env.step(action)
        
        return JSONResponse(
            content={
                "observation": obs.model_dump(mode="json"),
                "reward": {
                    "score": reward.score,
                    "breakdown": reward.breakdown.model_dump(mode="json"),
                    "feedback": reward.feedback,
                },
                "done": done,
                "info": info,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step_openenv")
async def step_openenv(request: StepRequest):
    """OpenEnv POST /step - generated_sql field."""
    action = SQLAction(query=request.generated_sql)
    obs, reward_float, done, info = env.step(action)
    return {
        "reward": reward_float,
        "done": done,
        "info": info
    }


@app.get("/state")
async def state():
    """Get current environment state."""
    s = env.state()
    return JSONResponse(content=s.model_dump(mode="json"))


@app.get("/questions")
async def get_questions():
    """List available questions for testing."""
    questions = {}
    for q_id, q_data in env._questions.items():
        questions[q_id] = {
            "text": q_data["text"],
            "task": q_data["task"],
            "difficulty": q_data["task"].split("_")[0],
        }
    return {"questions": questions}


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown."""
    env.close()
