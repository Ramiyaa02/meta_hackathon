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
from server.sql_query_environment import SQLQueryEnv


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

@app.get("/metadata")
async def metadata():
    """OpenEnv metadata."""
    return {
        "name": "sql_query_gen",
        "description": "SQL Query Generation OpenEnv",
        "version": "1.0.0"
    }

@app.get("/schema")
async def schema():
    """OpenEnv schemas."""
    from pydantic import BaseModel
    from models import SQLAction, SQLObservation, SQLState
    return {
        "action": SQLAction.model_json_schema(),
        "observation": SQLObservation.model_json_schema(),
        "state": SQLState.model_json_schema()
    }

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
    return {"status": "healthy"}


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
//openyaml
# OpenEnv Configuration - SQL Query Generation Environment
# Real-world task: Generate optimized SQL queries from natural language
#
# This environment trains LLMs to convert English descriptions into
# executable SQL queries with deterministic reward (correctness + efficiency + safety)

name: sql_query_gen
metadata:
  name: "sql_query_gen"
  version: "1.0.0"
  description: >
    A production RL environment where agents learn to generate SQL queries
    from natural language descriptions. The agent reads a text question,
    formulates a SQL query, and receives a reward based on:
    - Correctness (does it run? does it return right results?)
    - Efficiency (good query plan? uses indexes?)
    - Safety (guards against SQL injection? handles nulls?)
  author: "OpenEnv Contributors"
  license: "MIT"
  tags:
    - "nlp-to-sql"
    - "code-generation"
    - "optimization"
    - "real-world"
    - "deterministic-reward"
  homepage: "https://github.com/meta-pytorch/OpenEnv"

spec_version: "1.0.0"
spec_compliance: true

# Environment interface
environment:
  module: "server.sql_query_environment"
  class: "SQLQueryEnv"
  entry_point: "server.sql_query_environment:SQLQueryEnv"
  interface:
    - method: "reset"
      args: ["question_id (str, optional)"]
      returns: "SQLObservation"
      description: "Reset environment with a new question, return initial observation"
    - method: "step"
      args: ["action (SQLAction)"]
      returns: "(SQLObservation, reward (float), done (bool), info (dict))"
      description: "Submit SQL query, receive reward and environment feedback"
    - method: "state"
      returns: "dict"
      description: "Return current environment state"

# Task definitions
tasks:
  - id: "easy_select"
    name: "Simple SELECT"
    description: "Basic single-table query with WHERE clause"
    difficulty: "easy"
    max_turns: 1
    example_question: "Find all customers from New York"
    example_table: "customers (id, name, city, country)"
    reward_distribution: "Normal - 0.6-0.9 for good queries"

  - id: "medium_join"
    name: "Multi-table JOIN"
    description: "Query requiring 2-3 table joins with aggregation"
    difficulty: "medium"
    max_turns: 1
    example_question: "Show total sales per product category"
    example_tables: "orders, products, categories"
    reward_distribution: "Wide - 0.3-1.0 depending on join efficiency"

  - id: "hard_complex"
    name: "Complex Multi-Join"
    description: "Advanced query with subqueries, CTE, window functions"
    difficulty: "hard"
    max_turns: 1
    example_question: "Rank customers by total lifetime value with running average"
    example_tables: "customers, orders, products, order_items"
    reward_distribution: "Sparse - 0.0-1.0 (many wrong approaches)"

# Grading configuration
grading:
  metrics:
    - name: "correctness"
      weight: 0.50
      description: "Does the query execute and return correct results?"
      range: [0.0, 1.0]
    - name: "efficiency"
      weight: 0.30
      description: "Is the query optimized (uses indexes, minimal scans)?"
      range: [0.0, 1.0]
    - name: "safety"
      weight: 0.20
      description: "Is the query safe from injection and edge cases?"
      range: [0.0, 1.0]

# Reward function
reward:
  type: "weighted"
  formula: "0.5*correctness + 0.3*efficiency + 0.2*safety"
  range: [0.0, 1.0]
  partial_credit: true
  description: "Continuous reward signal for each query submission"

# Data configuration
data:
  database_engine: "sqlite"
  schema_location: "server/schema.sql"
  sample_data_location: "server/sample_data.sql"
  sample_questions_location: "server/sample_questions.json"
  tables:
    - name: "customers"
      columns: ["id", "name", "email", "city", "country", "created_at"]
    - name: "products"
      columns: ["id", "name", "category", "price", "stock"]
    - name: "orders"
      columns: ["id", "customer_id", "total", "order_date", "status"]
    - name: "order_items"
      columns: ["id", "order_id", "product_id", "quantity", "unit_price"]
    - name: "categories"
      columns: ["id", "name", "description"]

# Deployment
deployment:
  framework: "fastapi"
  base_image: "python:3.11-slim"
  dockerfile: "Dockerfile" # At project root
  port: 8000
  endpoints:
    - path: "/reset"
      method: "GET"
      params: ["question_id (optional)"]
      description: "Start new episode"
    - path: "/step"
      method: "POST"
      body: "SQLAction (query, reasoning)"
      description: "Submit query action"
    - path: "/state"
      method: "GET"
      description: "Get current state"
    - path: "/health"
      method: "GET"
      description: "Health check"
    - path: "/docs"
      method: "GET"
      description: "API documentation (Swagger UI)"
    - path: "/web"
      method: "GET"
      description: "Web interface"

# Hugging Face Spaces configuration
huggingface:
  sdk: "docker"
  app_port: 8000
  base_path: "/web"
  env_variables:
    - name: "HF_TOKEN"
      description: "Hugging Face API token"
      required: false
    - name: "DEMO_MODE"
      description: "Run with sample data only"
      default: "true"