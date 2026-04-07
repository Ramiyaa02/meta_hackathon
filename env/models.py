"""
Pydantic models for the OpenEnv AI Customer Support Ticket System.

Defines the Observation, Action, and Reward schemas used throughout
the environment for type safety and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AWAITING_CUSTOMER = "awaiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ActionType(str, Enum):
    """Allowed agent action types."""
    RESPOND = "respond"           # Send a text reply to the customer
    ESCALATE = "escalate"         # Escalate ticket to a senior agent
    RESOLVE = "resolve"           # Mark ticket as resolved
    REQUEST_INFO = "request_info" # Ask customer for more information
    REFUND = "refund"             # Issue a refund (if applicable)


# ---------------------------------------------------------------------------
# Observation — what the environment exposes to the agent each turn
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in the ticket conversation."""
    role: str = Field(..., description="'customer' or 'agent'")
    content: str = Field(..., description="Message text")
    timestamp: str = Field(..., description="ISO-8601 timestamp")


class TicketContext(BaseModel):
    """Full context of the current support ticket."""
    ticket_id: str
    subject: str
    priority: TicketPriority
    status: TicketStatus
    customer_name: str
    customer_email: str
    order_id: Optional[str] = None
    product_name: Optional[str] = None
    category: str = Field(..., description="e.g. billing, technical, order_status")
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """
    Observation returned by reset() and step().

    The agent receives the full ticket context plus an instruction
    describing what it should accomplish.
    """
    ticket: TicketContext
    instruction: str = Field(
        ...,
        description="Natural-language instruction telling the agent what to do"
    )
    available_actions: List[ActionType] = Field(
        default_factory=lambda: list(ActionType),
        description="Actions the agent may take this turn"
    )
    turn: int = Field(default=0, description="Current turn number")
    max_turns: int = Field(default=5, description="Maximum turns allowed")


# ---------------------------------------------------------------------------
# Action — what the agent sends back to the environment
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Action submitted by the agent via step().

    The agent must pick an action type and supply a response message.
    """
    action_type: ActionType = Field(
        ...,
        description="The type of action to perform"
    )
    response: str = Field(
        ...,
        description="The text response to send to the customer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra data (e.g. refund amount)"
    )


# ---------------------------------------------------------------------------
# Reward — evaluation returned after each step
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed breakdown of the reward components."""
    correctness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Did the agent take the right action?"
    )
    politeness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Was the response professional and empathetic?"
    )
    completeness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Did the response address all aspects of the issue?"
    )
    penalty: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Penalty deductions (e.g. rude language, wrong action)"
    )


class Reward(BaseModel):
    """
    Reward returned after each step().

    Provides a continuous score between 0.0 and 1.0 with a detailed
    breakdown so the agent (or its developer) can understand what
    went well and what did not.
    """
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall reward score"
    )
    breakdown: RewardBreakdown
    feedback: str = Field(
        ...,
        description="Human-readable feedback on the agent's performance"
    )
