"""
SQL Query Generation Environment.

Implements the OpenEnv reset/step/state interface for training LLMs
to generate SQL queries from natural language descriptions.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    Reward,
    RewardBreakdown,
    SQLAction,
    SQLObservation,
    SQLState,
    StepResult,
)


class SQLQueryEnv:
    """SQL Query Generation Environment."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize the environment.
        
        Args:
            db_path: Path to SQLite database (default: in-memory)
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        
        # Episode state
        self._current_question_id: Optional[str] = None
        self._current_question: Optional[str] = None
        self._expected_result: Optional[Dict[str, Any]] = None
        self._submitted_query: Optional[str] = None
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        
        # Sample questions
        self._questions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database
        self._init_database()
        self._load_sample_questions()

    def _init_database(self) -> None:
        """Initialize the database with schema and sample data."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Create tables
        schema_sql = """
        -- Customers table
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            city TEXT,
            country TEXT,
            created_at TEXT
        );
        
        -- Products table
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        );
        
        -- Orders table
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL,
            order_date TEXT,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        
        -- Order items table
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER,
            unit_price REAL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        
        -- Categories table
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT
        );
        
        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city);
        CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
        CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
        CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);
        CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
        """
        
        self.cursor.executescript(schema_sql)
        
        # Insert sample data
        data_sql = """
        DELETE FROM customers;
        DELETE FROM products;
        DELETE FROM orders;
        DELETE FROM order_items;
        DELETE FROM categories;
        
        -- Sample categories
        INSERT INTO categories (id, name, description) VALUES
        (1, 'Electronics', 'Electronic devices and gadgets'),
        (2, 'Clothing', 'Apparel and fashion'),
        (3, 'Books', 'Physical and digital books');
        
        -- Sample products
        INSERT INTO products (id, name, category, price, stock) VALUES
        (1, 'Laptop Pro', 'Electronics', 999.99, 50),
        (2, 'USB-C Cable', 'Electronics', 19.99, 200),
        (3, 'T-Shirt', 'Clothing', 29.99, 100),
        (4, 'Python Book', 'Books', 49.99, 75);
        
        -- Sample customers
        INSERT INTO customers (id, name, email, city, country, created_at) VALUES
        (1, 'Alice Johnson', 'alice@example.com', 'New York', 'USA', '2023-01-15'),
        (2, 'Bob Smith', 'bob@example.com', 'San Francisco', 'USA', '2023-02-20'),
        (3, 'Carol Davis', 'carol@example.com', 'New York', 'USA', '2023-03-10'),
        (4, 'David Miller', 'david@example.com', 'Boston', 'USA', '2023-04-05'),
        (5, 'Eve Wilson', 'eve@example.com', 'London', 'UK', '2023-05-12');
        
        -- Sample orders
        INSERT INTO orders (id, customer_id, total, order_date, status) VALUES
        (1, 1, 999.99, '2024-01-10', 'delivered'),
        (2, 1, 49.99, '2024-01-15', 'delivered'),
        (3, 2, 49.99, '2024-01-20', 'pending'),
        (4, 3, 29.99, '2024-02-01', 'delivered'),
        (5, 4, 1049.98, '2024-02-05', 'shipped');
        
        -- Sample order items
        INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
        (1, 1, 1, 1, 999.99),
        (2, 2, 4, 1, 49.99),
        (3, 3, 4, 1, 49.99),
        (4, 4, 3, 1, 29.99),
        (5, 5, 1, 1, 999.99),
        (6, 5, 2, 5, 19.99);
        """
        
        self.cursor.executescript(data_sql)
        self.conn.commit()

    def _load_sample_questions(self) -> None:
        """Load sample questions for testing."""
        self._questions = {
            "q1": {
                "text": "Find all customers from New York",
                "task": "easy_select",
                "expected_columns": ["id", "name", "email", "city", "country", "created_at"],
                "min_rows": 2,
                "max_rows": 5,
            },
            "q2": {
                "text": "Show the total sales per product",
                "task": "medium_join",
                "expected_columns": ["product_id", "name", "total_quantity", "total_sales"],
                "min_rows": 1,
                "max_rows": 10,
            },
            "q3": {
                "text": "Rank customers by total lifetime value",
                "task": "hard_complex",
                "expected_columns": ["customer_id", "name", "lifetime_value", "rank"],
                "min_rows": 1,
                "max_rows": 10,
            },
            "q4": {
                "text": "List all orders from 2024 with customer details",
                "task": "medium_join",
                "expected_columns": ["order_id", "customer_name", "order_date", "total"],
                "min_rows": 1,
                "max_rows": 20,
            },
            "q5": {
                "text": "Find products that have never been ordered",
                "task": "medium_join",
                "expected_columns": ["product_id", "product_name", "category"],
                "min_rows": 0,
                "max_rows": 5,
            },
        }

    def reset(self, question_id: Optional[str] = None) -> SQLObservation:
        """Reset environment and return initial observation.
        
        Args:
            question_id: Optional ID to load specific question (defaults to q1)
        
        Returns:
            Initial observation with question and schema
        """
        if question_id is None:
            question_id = "q1"
        
        if question_id not in self._questions:
            raise ValueError(f"Unknown question_id: {question_id}")
        
        self._current_question_id = question_id
        question_data = self._questions[question_id]
        self._current_question = question_data["text"]
        self._expected_result = question_data
        
        # Reset episode state
        self._submitted_query = None
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        
        # Build schema description
        schema_desc = self._get_schema_description()
        
        # Get sample data
        sample = self._get_sample_data()
        
        return SQLObservation(
            question=self._current_question,
            database_schema=schema_desc,
            sample_data=sample,
            expected_columns=question_data.get("expected_columns", []),
            expected_row_count_approx=question_data.get("max_rows"),
        )

    def step(self, action: SQLAction) -> Tuple[SQLObservation, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: SQLAction with query and optional reasoning
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        query = action.query.strip()
        self._submitted_query = query
        self._step_count += 1
        
        # Grade the query
        reward_obj = self._grade_query(query)
        reward_score = reward_obj.score
        self._cumulative_reward += reward_score
        
        # Return: (observation, reward_obj_for_feedback, reward_float_for_rl, done, info)
        # Actually keep it simple: (observation, reward_float, done, info)
        # And put reward details in info and observation
        
        # Execute query to get results
        execution_info = self._execute_query(query)
        
        # Build new observation
        obs = SQLObservation(
            question=self._current_question,
            database_schema=self._get_schema_description(),
            query_feedback=reward_obj.feedback,
            execution_result=execution_info,
            actual_columns=execution_info.get("columns", []),
            actual_row_count=execution_info.get("row_count", 0),
        )
        
        # Episode is done after one step (single-turn task)
        self._done = True
        
        info = {
            "question_id": self._current_question_id,
            "step": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "breakdown": reward_obj.breakdown.model_dump(),
            "execution_time_ms": execution_info.get("execution_time_ms", 0),
            "query_valid": execution_info.get("success", False),
        }
        
        return obs, reward_score, self._done, info

    def state(self) -> SQLState:
        """Return current environment state."""
        return SQLState(
            question_id=self._current_question_id or "none",
            question=self._current_question or "",
            submitted_query=self._submitted_query,
            last_reward=self._cumulative_reward if self._step_count > 0 else None,
            is_done=self._done,
            step_count=self._step_count,
            cumulative_reward=self._cumulative_reward,
        )

    def _get_schema_description(self) -> str:
        """Get readable database schema description."""
        schema = """
DATABASE SCHEMA:

1. customers
   - id (INTEGER PRIMARY KEY)
   - name (TEXT)
   - email (TEXT UNIQUE)
   - city (TEXT)
   - country (TEXT)
   - created_at (TEXT)

2. products
   - id (INTEGER PRIMARY KEY)
   - name (TEXT)
   - category (TEXT)
   - price (REAL)
   - stock (INTEGER)

3. categories
   - id (INTEGER PRIMARY KEY)
   - name (TEXT UNIQUE)
   - description (TEXT)

4. orders
   - id (INTEGER PRIMARY KEY)
   - customer_id (INTEGER, FK to customers.id)
   - total (REAL)
   - order_date (TEXT)
   - status (TEXT)

5. order_items
   - id (INTEGER PRIMARY KEY)
   - order_id (INTEGER, FK to orders.id)
   - product_id (INTEGER, FK to products.id)
   - quantity (INTEGER)
   - unit_price (REAL)

INDEXES: ix_customers_city, ix_orders_customer_id, ix_order_items_order_id, ix_order_items_product_id
"""
        return schema.strip()

    def _get_sample_data(self) -> str:
        """Get sample of each table for context."""
        sample_rows = {}
        
        for table in ["customers", "products", "orders"]:
            try:
                self.cursor.execute(f"SELECT * FROM {table} LIMIT 2")
                rows = self.cursor.fetchall()
                if rows:
                    cols = [desc[0] for desc in self.cursor.description]
                    data_str = "\n".join(
                        f"  {dict(row)}" for row in rows
                    )
                    sample_rows[table] = f"{table}:\n{data_str}"
            except Exception:
                pass
        
        return "\n".join(sample_rows.values()) if sample_rows else ""

    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute the query and return results.
        
        Returns:
            Dict with success, columns, row_count, error, etc
        """
        import time
        
        try:
            # Sanitize and validate before execution
            if any(kw in query.upper() for kw in ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE"]):
                return {
                    "success": False,
                    "error": "Modification queries not allowed",
                    "columns": [],
                    "row_count": 0,
                    "execution_time_ms": 0,
                }
            
            start_time = time.time()
            self.cursor.execute(query)
            execution_time = (time.time() - start_time) * 1000
            
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
            
            return {
                "success": True,
                "columns": columns,
                "row_count": len(rows),
                "sample_data": [dict(row) for row in rows[:3]],
                "execution_time_ms": execution_time,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "row_count": 0,
                "execution_time_ms": 0,
            }

    def _grade_query(self, query: str) -> Reward:
        """Grade the query on three dimensions.
        
        Returns:
            Reward object with scores and feedback
        """
        # 1. CORRECTNESS (50%): Does it execute without error?
        correctness = 1.0 if self._execute_query(query)["success"] else 0.0
        
        # 2. EFFICIENCY (30%): Good query plan? Uses indexes?
        efficiency = self._score_efficiency(query)
        
        # 3. SAFETY (20%): SQL injection protection? NULL handling?
        safety = self._score_safety(query)
        
        # Apply penalties
        penalty = self._compute_penalty(query)
        
        # Weighted average
        raw_score = 0.5 * correctness + 0.3 * efficiency + 0.2 * safety
        final_score = max(0.0, min(1.0, raw_score - penalty))
        
        # Build feedback
        feedback_parts = []
        if correctness == 1.0:
            feedback_parts.append("✓ Query executes successfully")
        else:
            feedback_parts.append("✗ Query has syntax errors or execution issues")
        
        if efficiency >= 0.8:
            feedback_parts.append("✓ Good query plan (efficient)")
        elif efficiency >= 0.5:
            feedback_parts.append("⚠ Query is functional but could be optimized")
        else:
            feedback_parts.append("✗ Query is inefficient (missing indexes, unnecessary scans)")
        
        if safety >= 0.9:
            feedback_parts.append("✓ Query is safe from injection and edge cases")
        elif safety >= 0.7:
            feedback_parts.append("⚠ Query has minor safety concerns")
        else:
            feedback_parts.append("✗ Query may have safety issues")
        
        feedback = " | ".join(feedback_parts)
        
        return Reward(
            score=final_score,
            breakdown=RewardBreakdown(
                correctness=correctness,
                efficiency=efficiency,
                safety=safety,
                penalty=penalty,
            ),
            feedback=feedback,
        )

    def _score_efficiency(self, query: str) -> float:
        """Score query efficiency (30% of reward).
        
        Heuristics:
        - Using indexes: +0.2
        - Avoiding full scans: +0.2
        - Using appropriate joins: +0.2
        - Avoiding subqueries when joins would work: -0.1
        """
        score = 0.6  # Base score
        query_upper = query.upper()
        
        # Bonus for explicit joins (better than implicit)
        if "JOIN" in query_upper:
            score += 0.2
        
        # Bonus for WHERE clauses (using indexes)
        if "WHERE" in query_upper:
            score += 0.1
        
        # Penalty for subqueries (less efficient)
        if "SELECT" in query_upper and query_upper.count("SELECT") > 1:
            score -= 0.1
        
        # Limit result set
        if "LIMIT" in query_upper:
            score += 0.1
        
        return min(1.0, max(0.0, score))

    def _score_safety(self, query: str) -> float:
        """Score query safety (20% of reward).
        
        Heuristics:
        - No DROP/DELETE/INSERT/UPDATE: +0.5
        - Uses parameterized style (no string concat): +0.3
        - Handles NULL cases: +0.2
        """
        score = 0.0
        query_upper = query.upper()
        
        # Dangerous operations
        dangerous_ops = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER"]
        if not any(op in query_upper for op in dangerous_ops):
            score += 0.5
        else:
            return 0.0  # Fail safety if dangerous
        
        # Check for string concatenation (SQL injection risk)
        if "||" not in query and "CONCAT" not in query_upper:
            score += 0.3
        
        # COALESCE or IFNULL for NULL handling
        if "COALESCE" in query_upper or "IFNULL" in query_upper or "IS NULL" in query_upper:
            score += 0.2
        
        return min(1.0, max(0.0, score))

    def _compute_penalty(self, query: str) -> float:
        """Compute penalties for violations.
        
        Returns:
            Penalty value (0.0-1.0)
        """
        penalty = 0.0
        query_upper = query.upper()
        
        # Very verbose or overly complex
        if len(query) > 2000:
            penalty += 0.1
        
        # Multiple unnecessary subqueries
        if query_upper.count("SELECT") > 3:
            penalty += 0.1
        
        # Potential N+1 queries pattern
        if query_upper.count("WHERE") > 3:
            penalty += 0.05
        
        return min(1.0, penalty)

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()