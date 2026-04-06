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