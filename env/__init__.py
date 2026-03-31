"""
AI Customer Support Ticket System — OpenEnv Environment.

A realistic simulation environment where an AI agent interacts
with customer support tickets and is evaluated on correctness,
politeness, and completeness of its responses.
"""

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType, Observation, Reward
from env.tasks import TASKS, get_task, list_tasks

__all__ = [
    "CustomerSupportEnv",
    "Action",
    "ActionType",
    "Observation",
    "Reward",
    "TASKS",
    "get_task",
    "list_tasks",
]
