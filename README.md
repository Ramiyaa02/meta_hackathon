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

## Problem Motivation
SQL query generation is a critical task for data analysts and business intelligence. This environment enables RL training of LLMs to produce optimized, safe SQL queries, reducing manual effort and errors.

## Action & Observation Spaces
- **Action (`SQLAction`):** A SQL query string (e.g., `"SELECT * FROM customers WHERE city = 'New York'"`). Optional reasoning field.
- **Observation (`SQLObservation`):** Natural language question, database schema, sample data, expected columns, and after step: query feedback, execution results.
- **State (`SQLState`):** Current question ID, submitted query, last reward, step count, cumulative reward, done flag.

## Tasks (Difficulty Progression)
| Task ID | Difficulty | Description | Expected Reward (good query) |
|---------|------------|-------------|------------------------------|
| q1 | Easy | Find all customers from New York (single table, WHERE) | 0.87 |
| q2 | Medium | Show total sales per product (JOIN + aggregation) | 0.90 |
| q3 | Medium | Rank customers by total lifetime value (multi-table) | 0.84 |
| q4 | Medium | List orders from 2024 with customer details | 0.87 |
| q5 | Hard | Find products never ordered (subquery / anti-join) | 0.80 |

## Reward Function (3‑axis)
- **Correctness (50%):** Query executes without errors and returns correct results.
- **Efficiency (30%):** Uses indexes, avoids full scans, proper JOINs.
- **Safety (20%):** No SQL injection, handles NULL values, no destructive commands.
Reward = 0.5×correctness + 0.3×efficiency + 0.2×safety, clamped to [0,1].

## Setup & Usage
### Local testing
```bash
pip install -r requirements.txt
python client.py # interactive demo
uvicorn app:app --reload # start server at http://localhost:7860
Inference (baseline)
Bashexport HF_TOKEN=your_hf_token
export ENV_URL=https://ramiyaa-sql_query_gen.hf.space
python inference.py
Docker
Bashdocker build -t sql_query_gen .
docker run -p 8000:8000 sql_query_gen
Baseline Scores
Using Qwen/Qwen2.5-72B-Instruct via HF router:
TaskAverage Rewardq10.87q20.90q30.84q40.87q50.80Overall0.86
API Endpoints

POST /reset – start new episode
POST /step_openenv – submit SQL query
GET /state – current environment state
GET /health – health check
GET /docs – Swagger UI

License
MIT
