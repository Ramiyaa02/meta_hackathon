---
title: SQL Query Generation
emoji: 🐘
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
---

# SQL Query Generation Environment

A real‑world RL environment where agents learn to generate SQL queries from natural language.

- **Tasks:** Simple SELECT, multi‑table JOIN, complex aggregations
- **Reward:** 3‑axis (correctness 50%, efficiency 30%, safety 20%)
- **API:** POST `/reset`, POST `/step_openenv`, GET `/state`

## Endpoints

- `POST /reset` – start a new episode (random question)
- `POST /step_openenv` – submit SQL query, get reward
- `GET /health` – health check

## Inference

Run `inference.py` with `HF_TOKEN` set to evaluate the baseline model.