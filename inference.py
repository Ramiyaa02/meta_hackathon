#!/usr/bin/env python3
"""
OpenEnv‑compliant inference script for SQL Query Generation.
Uses OpenAI client and outputs [START]/[STEP]/[END] logs.
"""

import os
import sys
import asyncio
from typing import List, Optional
import httpx
from openai import OpenAI

# ----------------------------------------------------------------------
# Environment variables (must be set by the evaluator)
# ----------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set", file=sys.stderr)
    sys.exit(1)

ENV_URL = os.getenv("ENV_URL", "https://ramiyaa-sql_query_gen.hf.space")
BENCHMARK = "sql-query-gen"
MAX_STEPS = 1
MAX_TOTAL_REWARD = 1.0

TASKS = [
    {"id": "q1", "text": "Find all customers from New York"},
    {"id": "q2", "text": "Show the total sales per product"},
    {"id": "q3", "text": "Rank customers by total lifetime value"},
]

# ----------------------------------------------------------------------
# OpenAI client (required)
# ----------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ----------------------------------------------------------------------
# Environment API calls
# ----------------------------------------------------------------------
async def reset_env():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{ENV_URL}/reset")
        resp.raise_for_status()
        return resp.json()

async def step_env(sql: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{ENV_URL}/step_openenv", json={"generated_sql": sql})
        resp.raise_for_status()
        return resp.json()

# ----------------------------------------------------------------------
# Generate SQL using OpenAI client
# ----------------------------------------------------------------------
def generate_sql(question: str) -> str:
    prompt = f"""Convert this natural language query to SQL. Output only the SQL statement, no explanation.
Natural language: {question}
SQL:"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
        )
        sql = completion.choices[0].message.content.strip()
        # Simple extraction if model adds extra text
        if "SELECT" not in sql.upper():
            import re
            match = re.search(r"(SELECT\s+.*?)(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
        return sql if sql else "SELECT 1"
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", file=sys.stderr)
        return "SELECT * FROM customers LIMIT 1"

# ----------------------------------------------------------------------
# Logging functions (required format)
# ----------------------------------------------------------------------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    error_str = error if error else "null"
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = str(success).lower()
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
async def main():
    for task in TASKS:
        task_id = task["id"]
        question = task["text"]

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            await reset_env()
            sql = generate_sql(question)
            step_result = await step_env(sql)
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", True)
            error = None

            log_step(step=1, action=sql, reward=reward, done=done, error=error)
            log_end(success=True, steps=1, score=reward, rewards=[reward])
        except Exception as e:
            log_step(step=1, action="", reward=0.0, done=True, error=str(e))
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])

if __name__ == "__main__":
    asyncio.run(main())