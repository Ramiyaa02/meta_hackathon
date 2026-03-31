#!/usr/bin/env python3
"""
Baseline inference script for the AI Customer Support Ticket System.

Uses the OpenAI Python client to drive an LLM agent through all
available tasks and reports the average score.

Environment variables:
    OPENAI_API_KEY  — API key for the LLM provider
    API_BASE_URL    — Base URL for the API (default: https://api.openai.com/v1)
    MODEL_NAME      — Model to use (default: gpt-4o-mini)

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from openai import OpenAI

from env import CustomerSupportEnv
from env.models import Action, ActionType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_IDS = [
    "easy_order_status",
    "medium_angry_refund",
    "hard_multi_issue",
]

SYSTEM_PROMPT = """\
You are an expert AI customer support agent. You will receive a support ticket
with full context (customer messages, order details, metadata) and an
instruction describing what you should accomplish.

Your job:
1. Read the ticket carefully.
2. Understand the customer's issue and emotional state.
3. Choose the most appropriate action type from the available options.
4. Craft a professional, empathetic, and complete response.

Respond ONLY with a JSON object in this exact format:
{
  "action_type": "<one of: respond, escalate, resolve, request_info, refund>",
  "response": "<your text response to the customer>"
}

Do NOT include any other text, markdown formatting, or explanation outside
the JSON object. The JSON must be parseable directly.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ticket_prompt(obs) -> str:
    """Build a human-readable prompt from the observation."""
    ticket = obs.ticket
    lines = [
        "=" * 60,
        f"TICKET: {ticket.ticket_id}",
        f"Subject: {ticket.subject}",
        f"Customer: {ticket.customer_name} ({ticket.customer_email})",
        f"Priority: {ticket.priority.value.upper()}",
        f"Status: {ticket.status.value}",
        f"Category: {ticket.category}",
    ]
    if ticket.order_id:
        lines.append(f"Order ID: {ticket.order_id}")
    if ticket.product_name:
        lines.append(f"Product: {ticket.product_name}")
    lines.append("")
    lines.append("CONVERSATION HISTORY:")
    lines.append("-" * 40)
    for msg in ticket.messages:
        lines.append(f"[{msg.role.upper()}] ({msg.timestamp})")
        lines.append(msg.content)
        lines.append("")
    lines.append("TICKET METADATA:")
    lines.append("-" * 40)
    lines.append(json.dumps(ticket.metadata, indent=2))
    lines.append("")
    lines.append("AVAILABLE ACTIONS:")
    lines.append(", ".join(a.value for a in obs.available_actions))
    lines.append("")
    lines.append(f"INSTRUCTION: {obs.instruction}")
    lines.append(f"Turn: {obs.turn + 1} / {obs.max_turns}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _parse_action(raw: str) -> Action:
    """Parse the LLM's JSON response into an Action object."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json ... ``|)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text.strip())

    action_type_str = data.get("action_type", "respond").lower()
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.RESPOND

    return Action(
        action_type=action_type,
        response=data.get("response", ""),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_task(
    client: OpenAI,
    env: CustomerSupportEnv,
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one task and return the results."""
    obs = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_id}")
        print(f"{'=' * 60}")

    step_results = []
    done = False

    while not done:
        prompt = _build_ticket_prompt(obs)

        if verbose:
            print(f"\n--- Turn {obs.turn + 1}/{obs.max_turns} ---")

        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        raw_output = response.choices[0].message.content or "{}"

        if verbose:
            print(f"LLM Output: {raw_output[:200]}...")

        # Parse action
        try:
            action = _parse_action(raw_output)
        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"Parse error: {e}. Falling back to RESPOND.")
            action = Action(
                action_type=ActionType.RESPOND,
                response=raw_output,
            )

        # Step the environment
        obs, reward, done, info = env.step(action)

        step_results.append({
            "turn": info["turn"],
            "action_type": action.action_type.value,
            "reward_score": reward.score,
            "correctness": reward.breakdown.correctness,
            "politeness": reward.breakdown.politeness,
            "completeness": reward.breakdown.completeness,
            "penalty": reward.breakdown.penalty,
        })

        if verbose:
            print(f"Reward: {reward.score:.4f}")
            print(f"  Correctness:  {reward.breakdown.correctness:.4f}")
            print(f"  Politeness:   {reward.breakdown.politeness:.4f}")
            print(f"  Completeness: {reward.breakdown.completeness:.4f}")
            print(f"  Penalty:      {reward.breakdown.penalty:.4f}")

    avg_score = env.average_reward

    if verbose:
        print(f"\nTask '{task_id}' complete. Average reward: {avg_score:.4f}")

    return {
        "task_id": task_id,
        "steps": step_results,
        "average_reward": avg_score,
    }


def main() -> None:
    """Run all tasks and print a summary."""
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Set it before running this script:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    env = CustomerSupportEnv()

    print("=" * 60)
    print("AI Customer Support — Baseline Inference")
    print(f"Model: {MODEL}")
    print(f"API Base: {API_BASE}")
    print(f"Tasks: {len(TASK_IDS)}")
    print("=" * 60)

    all_results = []
    for task_id in TASK_IDS:
        result = run_single_task(client, env, task_id, verbose=True)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    total_avg = 0.0
    for r in all_results:
        print(f"  {r['task_id']:30s}  avg_reward = {r['average_reward']:.4f}")
        total_avg += r["average_reward"]

    overall_avg = total_avg / len(all_results)
    print(f"\n  {'OVERALL AVERAGE':30s}  score = {overall_avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
