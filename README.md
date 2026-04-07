---
title: SQL Query Generation
emoji: 🐘
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# SQL Query Generation Environment

A real‑world RL environment where agents learn to generate SQL queries from natural language descriptions. The agent reads a business question, writes an SQL query, and receives a deterministic reward based on correctness, efficiency, and safety.

## Why SQL Generation for RL?

SQL is the language of data. Teaching LLMs to write correct, fast, and safe queries reduces manual data analysis work. This environment provides a realistic training ground with a dense reward signal – exactly what RL needs. The multi‑turn design allows agents to refine queries based on feedback, mirroring real‑world SQL development.

## Problem Motivation

SQL query generation is a critical task for data analysts and business intelligence. This environment enables RL training of LLMs to produce optimized, safe SQL queries, reducing manual effort and errors.

## Action & Observation Spaces

- **Action (`SQLAction`):** A SQL query string (e.g., `"SELECT * FROM customers WHERE city = 'New York'"`). Optional reasoning field.
- **Observation (`SQLObservation`):** Natural language question, database schema, sample data, expected columns, and after step: query feedback, execution results.
- **State (`SQLState`):** Current question ID, submitted query, last reward, step count, cumulative reward, done flag.

## Tasks (Difficulty Progression)

| Task ID | Difficulty | Description | Baseline Reward (Qwen2.5-72B) |
|---------|------------|-------------|-------------------------------|
| q1 | Easy | Find all customers from New York (single table, WHERE) | 0.87 |
| q2 | Medium | Show total sales per product (JOIN + aggregation) | 0.90 |
| q3 | Medium | Rank customers by total lifetime value (multi-table) | 0.84 |
| q4 | Medium | List orders from 2024 with customer details | 0.87 |
| q5 | Hard | Find products never ordered (subquery / anti-join) | 0.80 |
| q6 | Hard | Running total of sales per customer (window function) | 0.75 |
| q7 | Hard | Customers who ordered every product category (relational division) | 0.70 |
| q8 | Hard | 7‑day moving average of daily sales | 0.78 |

## Reward Function (3‑axis + semantic correctness)

- **Correctness (50%):** Query executes without errors **and** returns the expected columns/rows (blends execution success with semantic column/row matching).
- **Efficiency (30%):** Uses indexes, avoids full scans, proper JOINs, no `SELECT *`, penalises slow execution.
- **Safety (20%):** No SQL injection, handles NULL values, no destructive commands.

Reward = 0.5×correctness + 0.3×efficiency + 0.2×safety, clamped to [0,1].

## Setup & Usage

### Local testing

```bash
pip install -r requirements.txt
python client.py          # interactive demo
uvicorn app:app --reload  # start server at http://localhost:8000
Inference (baseline)
Set environment variables and run:

bash
export HF_TOKEN=your_token
export ENV_URL=https://ramiyaa-sql-query-gen.hf.space
python inference.py
Docker
bash
docker build -t sql_query_gen .
docker run -p 7860:7860 sql_query_gen
Baseline Scores
Using Qwen/Qwen2.5-72B-Instruct via HF router:

Task	Average Reward
q1	0.87
q2	0.90
q3	0.84
q4	0.87
q5	0.80
q6	0.75
q7	0.70
q8	0.78
Overall	0.81
API Endpoints

GET /health – health check
GET /reset?question_id=q1 – reset with specific question
POST /reset – reset with random question (OpenEnv spec)
POST /step – submit SQLAction (OpenAI‑compatible)
POST /step_openenv – submit {"generated_sql": "..."} (used by inference script)
GET /state – current environment state
GET /questions – list all tasks
GET /demo – Gradio UI (manual testing and visualisation)

License
MIT