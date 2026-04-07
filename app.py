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
import gradio as gr

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
# Endpoints (existing)
# ============================================================================

@app.get("/metadata")
async def metadata():
    return {
        "name": "sql_query_gen",
        "description": "SQL Query Generation OpenEnv",
        "version": "1.0.0"
    }

@app.get("/schema")
async def schema():
    from models import SQLAction, SQLObservation, SQLState
    return {
        "action": SQLAction.model_json_schema(),
        "observation": SQLObservation.model_json_schema(),
        "state": SQLState.model_json_schema()
    }

@app.get("/")
async def root():
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
            "GET  /demo", 
        ],
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/reset")
async def reset(question_id: str = "q1"):
    try:
        obs = env.reset(question_id=question_id)
        return JSONResponse(content=obs.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset_openenv():
    q_ids = ["q1", "q2", "q3", "q4", "q5","q6","q7","q8"]
    q_id = random.choice(q_ids)
    obs = env.reset(question_id=q_id)
    return obs.model_dump()

@app.post("/step")
async def step(action: SQLAction):
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
    action = SQLAction(query=request.generated_sql)
    obs, reward_float, done, info = env.step(action)
    return {
        "reward": reward_float,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    s = env.state()
    return JSONResponse(content=s.model_dump(mode="json"))

@app.get("/questions")
async def get_questions():
    questions = {}
    for q_id, q_data in env._questions.items():
        questions[q_id] = {
            "text": q_data["text"],
            "task": q_data["task"],
            "difficulty": q_data["task"].split("_")[0],
        }
    return {"questions": questions}


# ============================================================================
# Gradio UI (additional – does not break existing endpoints)
# ============================================================================

async def grade_query_ui(question_id: str, sql_query: str):
    """Gradio function to grade a SQL query."""
    if not sql_query.strip():
        return "⚠️ Please enter a SQL query."

    try:
        obs = env.reset(question_id=question_id)
        action = SQLAction(query=sql_query)
        obs, reward_score, done, info = env.step(action)
        breakdown = info.get("breakdown", {})
        feedback = obs.query_feedback or ""

        result = f"""
### Results for task `{question_id}`: {obs.question}

| Metric | Score |
|--------|-------|
| **Reward** | **{reward_score:.3f}** |
| Correctness | {breakdown.get('correctness', 0):.2f} |
| Efficiency | {breakdown.get('efficiency', 0):.2f} |
| Safety | {breakdown.get('safety', 0):.2f} |

**Feedback:** {feedback}

**Execution info:** {info.get('execution_time_ms', 0):.2f} ms, rows returned: {obs.actual_row_count or 0}
"""
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="SQL Query Grader", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐘 SQL Query Generation Environment")
    gr.Markdown("Select a task, write an SQL query, and see how well it performs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
                label="Task",
                value="q1"
            )
            task_desc = gr.Markdown("**Task description will appear here**")
        with gr.Column(scale=2):
            sql_input = gr.Textbox(
                label="SQL Query",
                placeholder="SELECT ...",
                lines=8,
                value="SELECT id, name, email, city FROM customers WHERE city = 'New York'"
            )
            submit_btn = gr.Button("Grade Query", variant="primary")
    output = gr.Markdown(label="Results")

    def update_task_desc(question_id):
        desc = env._questions.get(question_id, {}).get("text", "Unknown task")
        return f"**Task:** {desc}"
    
    task_dropdown.change(fn=update_task_desc, inputs=task_dropdown, outputs=task_desc)
    submit_btn.click(fn=grade_query_ui, inputs=[task_dropdown, sql_input], outputs=output)
    
    gr.Markdown("### Example Queries")
    gr.Examples(
        examples=[
            ["q1", "SELECT * FROM customers WHERE city = 'New York'"],
            ["q2", "SELECT p.name, SUM(oi.quantity * oi.unit_price) as total_sales FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.name"],
            ["q5", "SELECT p.id, p.name FROM products p WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.id)"],
        ],
        inputs=[task_dropdown, sql_input],
        label="Try these"
    )

# Mount Gradio app at path "/demo"
app = gr.mount_gradio_app(app, demo, path="/demo")

# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    env.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)