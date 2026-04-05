"""
Client for SQL Query Generation Environment.

Use this to test the environment locally without running the server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import SQLAction
from server.sql_query_environment import SQLQueryEnv


class SQLQueryGenClient:
    """Client wrapper for SQL Query Gen environment."""

    def __init__(self, db_path: str = ":memory:"):
        self.env = SQLQueryEnv(db_path=db_path)

    def reset(self, question_id: str = "q1"):
        """Reset environment and return observation."""
        return self.env.reset(question_id)

    def step(self, query: str, reasoning: str = ""):
        """Submit a query and get results."""
        action = SQLAction(query=query, reasoning=reasoning)
        obs, reward_score, done, info = self.env.step(action)
        return {
            "observation": obs,
            "reward_score": reward_score,
            "reward_breakdown": info.get("breakdown", {}),
            "feedback": obs.query_feedback,
            "done": done,
            "info": info,
        }

    def state(self):
        """Get current state."""
        return self.env.state()

    def close(self):
        """Close environment."""
        self.env.close()


def run_demo():
    """Run a demo episode."""
    print("=" * 70)
    print("SQL QUERY GENERATION ENVIRONMENT - INTERACTIVE DEMO")
    print("=" * 70)

    client = SQLQueryGenClient()
    
    # Demo 1: Easy query
    print("\n📝 DEMO 1: Easy Query (Find customers from New York)")
    print("-" * 70)
    
    obs = client.reset("q1")
    print(f"\n❓ Question: {obs.question}")
    print(f"\n📊 Schema:\n{obs.database_schema}")
    
    query1 = "SELECT id, name, email, city FROM customers WHERE city = 'New York'"
    print(f"\n🔍 Submitted Query:\n{query1}")
    
    result1 = client.step(query1)
    print(f"\n✅ Reward: {result1['reward_score']:.4f}")
    print(f"   Feedback: {result1['feedback']}")
    print(f"   Breakdown: {result1['reward_breakdown']}")
    
    # Demo 2: Medium query
    print("\n" + "=" * 70)
    print("\n📝 DEMO 2: Medium Query (Show total sales per product)")
    print("-" * 70)
    
    obs = client.reset("q2")
    print(f"\n❓ Question: {obs.question}")
    
    query2 = """
    SELECT 
        p.id as product_id,
        p.name,
        SUM(oi.quantity) as total_quantity,
        SUM(oi.quantity * oi.unit_price) as total_sales
    FROM products p
    LEFT JOIN order_items oi ON p.id = oi.product_id
    GROUP BY p.id, p.name
    ORDER BY total_sales DESC
    """
    print(f"\n🔍 Submitted Query:\n{query2}")
    
    result2 = client.step(query2)
    print(f"\n✅ Reward: {result2['reward_score']:.4f}")
    print(f"   Feedback: {result2['feedback']}")
    
    # Demo 3: Inefficient query
    print("\n" + "=" * 70)
    print("\n📝 DEMO 3: Inefficient Query (Many subqueries)")
    print("-" * 70)
    
    obs = client.reset("q4")
    print(f"\n❓ Question: {obs.question}")
    
    # Bad query with subqueries instead of joins
    query3 = """
    SELECT 
        (SELECT o.id FROM orders o WHERE o.customer_id = c.id LIMIT 1) as order_id,
        c.name,
        (SELECT o.order_date FROM orders o WHERE o.customer_id = c.id LIMIT 1) as order_date,
        (SELECT o.total FROM orders o WHERE o.customer_id = c.id LIMIT 1) as total
    FROM customers c
    WHERE c.id IN (SELECT customer_id FROM orders WHERE YEAR(order_date) = 2024)
    """
    print(f"\n🔍 Submitted Query (with multiple subqueries):\n{query3}")
    
    result3 = client.step(query3)
    print(f"\n✅ Reward: {result3['reward_score']:.4f}")
    print(f"   Feedback: {result3['feedback']}")
    
    # Show state
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("-" * 70)
    state = client.state()
    print(f"\nQuestion ID: {state.question_id}")
    print(f"Steps taken: {state.step_count}")
    print(f"Cumulative reward: {state.cumulative_reward:.4f}")
    
    client.close()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    run_demo()
