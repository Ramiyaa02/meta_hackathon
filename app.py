"""
FastAPI server for SQL Query Generation OpenEnv.

Exposes the environment over HTTP for Hugging Face Spaces deployment
and programmatic access.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import random

from sql_query_environment import SQLQueryEnv
from models import SQLAction, SQLObservation, Reward, SQLState


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

class StepRequest(BaseModel):
    generated_sql: str


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
            "POST /step_openenv",
            "GET  /state",
            "GET  /health",
            "GET  /docs",
        ],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/reset")
async def reset(question_id: str = "q1"):
    """Reset environment with a specific question (GET)."""
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
    """Submit a SQL query and receive reward (OpenAI‑compatible)."""
    try:
        obs, reward_float, done, info = env.step(action)
        # Return the same structure as the original step endpoint
        return JSONResponse(
            content={
                "observation": obs.model_dump(mode="json"),
                "reward": reward_float,
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
        "info": info,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)