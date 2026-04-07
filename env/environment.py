"""
Main OpenEnv environment class for the AI Customer Support Ticket System.

Implements the standard OpenEnv interface:
  - reset()  -> Observation
  - step(action) -> (Observation, Reward, done, info)
  - state()  -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from env.graders import grade
from env.models import Action, Observation, Reward, TicketStatus
from env.tasks import TASKS, Task, get_task, list_tasks


class CustomerSupportEnv:
    """
    OpenEnv-compliant environment for AI customer support simulation.

    Usage:
        env = CustomerSupportEnv()
        obs = env.reset(task_id="easy_order_status")
        obs, reward, done, info = env.step(action)
        current_state = env.state()
    """

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._observation: Optional[Observation] = None
        self._turn: int = 0
        self._done: bool = False
        self._rewards: list[float] = []
        self._history: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy_order_status") -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            task_id: One of the registered task IDs (easy_order_status,
                     medium_angry_refund, hard_multi_issue).

        Returns:
            The initial Observation for the selected task.
        """
        self._task = get_task(task_id)
        self._observation = self._task.build_observation()
        self._turn = 0
        self._done = False
        self._rewards = []
        self._history = []

        # Update turn in observation
        self._observation.turn = self._turn
        return self._observation

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Submit an action and receive the next observation, reward, and status.

        Args:
            action: The agent's Action (action_type + response text).

        Returns:
            A tuple of (observation, reward, done, info).
        """
        if self._task is None or self._observation is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() before step()."
            )

        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        # Grade the action
        reward = grade(action, self._observation, self._task)
        self._rewards.append(reward.score)

        # Record history
        self._history.append({
            "turn": self._turn,
            "action_type": action.action_type.value,
            "response": action.response,
            "reward_score": reward.score,
            "reward_breakdown": reward.breakdown.model_dump(),
        })

        # Advance turn
        self._turn += 1

        # Determine if done
        max_turns = self._task.max_turns
        is_last_turn = self._turn >= max_turns
        is_resolved = action.action_type.value in ("resolve",)
        self._done = is_last_turn or is_resolved

        # Build info dict
        info: Dict[str, Any] = {
            "turn": self._turn,
            "max_turns": max_turns,
            "done_reason": (
                "resolved" if is_resolved
                else "max_turns_reached" if is_last_turn
                else "continue"
            ),
            "cumulative_score": round(sum(self._rewards) / len(self._rewards), 4),
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
        }

        # Build next observation
        if not self._done:
            # Add the agent's response to the conversation
            from env.models import Message

            agent_message = Message(
                role="agent",
                content=action.response,
                timestamp="2025-06-10T18:00:00Z",
            )
            self._observation.ticket.messages.append(agent_message)
            self._observation.turn = self._turn

            # Update ticket status based on action
            if action.action_type.value == "escalate":
                self._observation.ticket.status = TicketStatus.IN_PROGRESS
            elif action.action_type.value == "refund":
                self._observation.ticket.status = TicketStatus.IN_PROGRESS
            elif action.action_type.value == "request_info":
                self._observation.ticket.status = TicketStatus.AWAITING_CUSTOMER
            else:
                self._observation.ticket.status = TicketStatus.IN_PROGRESS
        else:
            # Mark ticket as resolved when done
            if is_resolved:
                self._observation.ticket.status = TicketStatus.RESOLVED
            self._observation.turn = self._turn

        return self._observation, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state as a dictionary.

        Useful for debugging, logging, or external inspection.
        """
        if self._task is None or self._observation is None:
            return {"status": "not_initialized"}

        return {
            "status": "active" if not self._done else "done",
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "turn": self._turn,
            "max_turns": self._task.max_turns,
            "ticket": self._observation.ticket.model_dump(),
            "cumulative_reward": (
                round(sum(self._rewards) / len(self._rewards), 4)
                if self._rewards
                else 0.0
            ),
            "history": self._history,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def current_task(self) -> Optional[Task]:
        return self._task

    @property
    def average_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return round(sum(self._rewards) / len(self._rewards), 4)

    @staticmethod
    def available_tasks() -> list:
        """Return metadata for all available tasks."""
        return list_tasks()
