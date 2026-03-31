# AI Customer Support Ticket System — OpenEnv

A production-ready [OpenEnv](https://openenv.dev) simulation environment where an AI agent handles real-world customer support tickets. The agent reads ticket context, selects an appropriate action, and crafts a professional response — then receives a continuous reward score evaluating **correctness**, **politeness**, and **completeness**.

---

## Real-World Motivation

Customer support is one of the highest-volume knowledge-work tasks in the world. Companies receive thousands of tickets daily, and response quality directly impacts customer retention, NPS scores, and brand reputation. This environment simulates that workflow so that AI agents can be trained and evaluated on:

- Understanding customer intent and emotional state
- Choosing the right resolution action (respond, refund, escalate, resolve)
- Writing empathetic, complete, and professional responses
- Handling multi-issue tickets with competing priorities

---

## Action & Observation Space

### Observation

Each turn the agent receives an `Observation` containing:

| Field               | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `ticket`             | Full ticket context: messages, metadata, priority, status, etc.   |
| `instruction`        | Natural-language instruction telling the agent what to accomplish  |
| `available_actions`  | List of action types the agent may choose this turn                |
| `turn`               | Current turn number (0-indexed)                                    |
| `max_turns`          | Maximum turns allowed for this task                                |

### Action

The agent returns an `Action` with:

| Field          | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `action_type`   | One of: `respond`, `escalate`, `resolve`, `request_info`, `refund` |
| `response`      | The text response to send to the customer                          |
| `metadata`      | Optional extra data (e.g., refund amount)                          |

### Reward

After each step the agent receives a `Reward` with:

| Field        | Description                                              |
|---------------|----------------------------------------------------------|
| `score`       | Overall continuous score in [0.0, 1.0]                   |
| `breakdown`   | Individual scores for correctness, politeness, completeness, penalty |
| `feedback`    | Human-readable explanation of the evaluation             |

---

## Tasks

### Easy — Order Status Query (`easy_order_status`)

A customer asks about their order. The agent must look up tracking info in the ticket metadata and provide a friendly, helpful response with delivery details.

- **Max turns:** 3
- **Key skills:** Information retrieval, friendly tone

### Medium — Angry Customer Requesting Refund (`medium_angry_refund`)

A customer is furious about a defective product and demands a refund. The agent must de-escalate with empathy, apologize, and process the refund.

- **Max turns:** 5
- **Key skills:** De-escalation, empathy, refund processing

### Hard — Multi-Issue Ticket (`hard_multi_issue`)

A loyal customer has both a billing issue (double charge) AND a technical issue (app sync error). The agent must address BOTH problems comprehensively in a single coherent response.

- **Max turns:** 5
- **Key skills:** Multi-issue handling, prioritization, technical troubleshooting

---

## Project Structure

```
├── env/
│   ├── __init__.py          # Package init, exports
│   ├── models.py            # Pydantic models (Observation, Action, Reward)
│   ├── environment.py       # OpenEnv environment class (reset/step/state)
│   ├── tasks.py             # Task definitions (easy/medium/hard)
│   └── graders.py           # Deterministic grading functions
├── data/
│   └── tickets.json         # Realistic sample ticket data
├── openenv.yaml             # OpenEnv metadata & configuration
├── inference.py             # Baseline inference script (OpenAI client)
├── app.py                   # FastAPI server for HF Spaces
├── Dockerfile               # Docker build for deployment
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 3. Test with curl

```bash
# Reset environment
curl "http://localhost:8000/reset?task_id=easy_order_status"

# Submit an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "response": "Hello Alice! Your order has been shipped via UPS with tracking number 1Z999AA10123456784. Expected delivery is June 12. Let me know if you need anything else!"
  }'

# Check state
curl http://localhost:8000/state
```

---

## How to Run inference.py

The baseline script uses the OpenAI API to drive an LLM agent through all three tasks.

```bash
# Set environment variables
export OPENAI_API_KEY="sk-your-key-here"
export API_BASE_URL="https://api.openai.com/v1"   # or your provider's URL
export MODEL_NAME="gpt-4o-mini"                    # or gpt-4o, etc.

# Run
python inference.py
```

The script will:
1. Reset the environment for each task
2. Send the ticket context to the LLM
3. Parse the LLM's JSON response into an Action
4. Step the environment and collect the reward
5. Print per-step and per-task scores
6. Output the final average score across all tasks

---

## How to Run Docker

```bash
# Build
docker build -t ai-customer-support .

# Run (server only)
docker run -p 8000:8000 ai-customer-support

# Run with API key for inference
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-your-key" \
  -e MODEL_NAME="gpt-4o-mini" \
  ai-customer-support
```

---

## Hugging Face Spaces Deployment

This project is structured for direct deployment to Hugging Face Spaces (Docker mode):

1. Create a new Space with **Docker** SDK
2. Upload all files preserving the directory structure
3. The Space will automatically build and start the FastAPI server on port 8000
4. Set `OPENAI_API_KEY` as a Space secret for inference

---

## Expected Baseline Scores

Using `gpt-4o-mini` with the default grading system:

| Task                         | Expected Avg Score |
|------------------------------|--------------------|
| Easy (Order Status)          | 0.70 – 0.85        |
| Medium (Angry Refund)        | 0.60 – 0.80        |
| Hard (Multi-Issue)           | 0.50 – 0.75        |
| **Overall Average**          | **0.60 – 0.80**    |

Scores vary based on the model's ability to:
- Select the correct action type (correctness)
- Use empathetic, professional language (politeness)
- Cover all required topics and keywords (completeness)
- Avoid forbidden phrases and bad practices (penalties)

A perfect score of 1.0 requires the agent to take the optimal action, use highly empathetic language, and address every aspect of the customer's issue.

---

## Grading System

The reward function evaluates three dimensions:

| Dimension      | Weight | What It Measures                                         |
|----------------|--------|----------------------------------------------------------|
| Correctness    | 35%    | Was the right action type chosen?                        |
| Politeness     | 30%    | Was the response empathetic and professional?            |
| Completeness   | 35%    | Were all expected keywords/topics covered?               |
| Penalty        | —      | Deductions for forbidden phrases, empty responses, etc.  |

Final score = `0.35 × correctness + 30 × politeness + 0.35 × completeness − penalty`

---

## License

MIT
