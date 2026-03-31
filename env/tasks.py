"""
Task definitions for the AI Customer Support Ticket System.

Each task represents a realistic customer-support scenario with
varying difficulty. Tasks provide the initial ticket context,
the instruction for the agent, and the expected resolution criteria
used by graders.
"""

from __future__ import annotations

from typing import Any, Dict, List

from env.models import (
    ActionType,
    Message,
    Observation,
    TicketContext,
    TicketPriority,
    TicketStatus,
)


# ---------------------------------------------------------------------------
# Task base & concrete tasks
# ---------------------------------------------------------------------------

class Task:
    """Base class for all tasks."""

    task_id: str = ""
    difficulty: str = ""
    description: str = ""
    max_turns: int = 5

    def build_observation(self) -> Observation:
        raise NotImplementedError

    def expected_actions(self) -> List[ActionType]:
        """Actions the agent is expected to take (in order)."""
        raise NotImplementedError

    def expected_keywords(self) -> List[str]:
        """Keywords that should appear in the agent's response."""
        raise NotImplementedError

    def forbidden_phrases(self) -> List[str]:
        """Phrases that should NOT appear in the response."""
        return []


class EasyTask(Task):
    """
    EASY — Order Status Query

    Customer simply asks about the status of their order.
    The agent should look up the order, provide tracking info,
    and close the ticket or mark it resolved.
    """

    task_id = "easy_order_status"
    difficulty = "easy"
    description = "Respond to a simple order status inquiry."
    max_turns = 3

    def build_observation(self) -> Observation:
        ticket = TicketContext(
            ticket_id="TKT-1001",
            subject="Where is my order?",
            priority=TicketPriority.LOW,
            status=TicketStatus.OPEN,
            customer_name="Alice Johnson",
            customer_email="alice@example.com",
            order_id="ORD-88432",
            product_name="Wireless Bluetooth Headphones",
            category="order_status",
            messages=[
                Message(
                    role="customer",
                    content=(
                        "Hi, I placed an order (ORD-88432) for Wireless Bluetooth "
                        "Headphones about a week ago and I still haven't received it. "
                        "Can you please check the status? Thanks!"
                    ),
                    timestamp="2025-06-10T09:15:00Z",
                ),
            ],
            metadata={
                "order_status": "shipped",
                "tracking_number": "1Z999AA10123456784",
                "carrier": "UPS",
                "estimated_delivery": "2025-06-12",
                "shipped_date": "2025-06-07",
            },
        )
        return Observation(
            ticket=ticket,
            instruction=(
                "The customer wants to know the status of their order. "
                "Look up the order details in the ticket metadata and provide "
                "a helpful, friendly response with the tracking information."
            ),
            available_actions=[ActionType.RESPOND, ActionType.REQUEST_INFO, ActionType.RESOLVE],
            turn=0,
            max_turns=self.max_turns,
        )

    def expected_actions(self) -> List[ActionType]:
        return [ActionType.RESPOND, ActionType.RESOLVE]

    def expected_keywords(self) -> List[str]:
        return ["shipped", "tracking", "UPS", "1Z999AA10123456784", "June 12"]

    def forbidden_phrases(self) -> List[str]:
        return ["I don't know", "not my problem", "figure it out"]


class MediumTask(Task):
    """
    MEDIUM — Angry Customer Requesting Refund

    Customer is upset about a defective product and demands a refund.
    The agent must de-escalate, show empathy, and process the refund
    or offer an alternative resolution.
    """

    task_id = "medium_angry_refund"
    difficulty = "medium"
    description = "Handle an angry customer demanding a refund for a defective product."
    max_turns = 5

    def build_observation(self) -> Observation:
        ticket = TicketContext(
            ticket_id="TKT-2045",
            subject="DEFECTIVE PRODUCT - I want my money back NOW!",
            priority=TicketPriority.HIGH,
            status=TicketStatus.OPEN,
            customer_name="Robert Chen",
            customer_email="r.chen@example.com",
            order_id="ORD-77210",
            product_name="Smart Home Security Camera",
            category="billing",
            messages=[
                Message(
                    role="customer",
                    content=(
                        "I am absolutely furious! I bought your Smart Home Security "
                        "Camera (ORD-77210) two weeks ago and it's COMPLETELY broken. "
                        "The video feed freezes every 30 seconds, the night vision "
                        "doesn't work at all, and the mobile app crashes constantly. "
                        "This is unacceptable for a $199 product! I want a FULL refund "
                        "immediately or I'm filing a chargeback and leaving negative "
                        "reviews everywhere. This is the worst product I've ever purchased."
                    ),
                    timestamp="2025-06-10T14:30:00Z",
                ),
                Message(
                    role="customer",
                    content=(
                        "I've been waiting for 2 hours for a response! Are you even "
                        "reading my messages?! I'm done waiting. REFUND. NOW."
                    ),
                    timestamp="2025-06-10T16:45:00Z",
                ),
            ],
            metadata={
                "purchase_date": "2025-05-27",
                "price_paid": 199.99,
                "warranty_status": "active",
                "return_eligible": True,
                "previous_tickets": 0,
            },
        )
        return Observation(
            ticket=ticket,
            instruction=(
                "The customer is angry about a defective product and demanding a refund. "
                "You must: 1) Acknowledge their frustration with empathy, 2) Apologize "
                "for the inconvenience, 3) Offer a resolution (refund or replacement). "
                "The order is within the return window and the customer is eligible for "
                "a full refund."
            ),
            available_actions=[
                ActionType.RESPOND,
                ActionType.REFUND,
                ActionType.ESCALATE,
                ActionType.RESOLVE,
            ],
            turn=0,
            max_turns=self.max_turns,
        )

    def expected_actions(self) -> List[ActionType]:
        return [ActionType.RESPOND, ActionType.REFUND, ActionType.RESOLVE]

    def expected_keywords(self) -> List[str]:
        return [
            "sorry", "apologize", "understand", "frustrat",
            "refund", "full refund", "$199.99",
        ]

    def forbidden_phrases(self) -> List[str]:
        return [
            "calm down",
            "not our fault",
            "you should have",
            "that's your problem",
            "we don't care",
        ]


class HardTask(Task):
    """
    HARD — Multi-Issue Ticket (Billing + Technical)

    Customer has both a billing discrepancy (double charge) AND a
    technical issue (app not syncing). The agent must address both
    issues, prioritize correctly, and provide a comprehensive resolution.
    """

    task_id = "hard_multi_issue"
    difficulty = "hard"
    description = "Resolve a complex ticket involving both billing and technical issues."
    max_turns = 5

    def build_observation(self) -> Observation:
        ticket = TicketContext(
            ticket_id="TKT-3078",
            subject="Double charged AND app not working — need help ASAP",
            priority=TicketPriority.CRITICAL,
            status=TicketStatus.OPEN,
            customer_name="Maria Garcia",
            customer_email="m.garcia@example.com",
            order_id="ORD-99102",
            product_name="Premium Cloud Storage Subscription",
            category="billing",
            messages=[
                Message(
                    role="customer",
                    content=(
                        "Hello, I have TWO serious issues that need immediate attention. "
                        "\n\n1) BILLING ISSUE: I was charged $49.99 TWICE on June 5th for "
                        "my Premium Cloud Storage subscription (ORD-99102). My bank "
                        "statement clearly shows two identical charges. I need one of "
                        "these charges reversed ASAP — that's almost $100 taken from my "
                        "account incorrectly!\n\n"
                        "2) TECHNICAL ISSUE: On top of that, your desktop app (v3.2.1) "
                        "has completely stopped syncing my files since last week. I have "
                        "important work documents that I need access to and the app just "
                        "shows 'Sync Failed — Error Code 503' every time. I've tried "
                        "restarting the app and my computer multiple times.\n\n"
                        "I've been a loyal customer for 3 years and this is completely "
                        "unacceptable. Please fix BOTH issues. I expect a response today."
                    ),
                    timestamp="2025-06-10T08:00:00Z",
                ),
                Message(
                    role="customer",
                    content=(
                        "It's been 6 hours. I'm now considering switching to a competitor. "
                        "Please address both the billing and the sync issue."
                    ),
                    timestamp="2025-06-10T14:00:00Z",
                ),
                Message(
                    role="customer",
                    content=(
                        "I also want to mention that I tried reinstalling the app but the "
                        "sync issue persists. The error log shows: "
                        "'[ERROR] SyncEngine: connection timeout after 30s — server "
                        "unreachable at api.cloudstore.example.com'. "
                        "Please look into this."
                    ),
                    timestamp="2025-06-10T15:30:00Z",
                ),
            ],
            metadata={
                "billing": {
                    "charge_amount": 49.99,
                    "charge_date": "2025-06-05",
                    "charge_count": 2,
                    "subscription_plan": "Premium Cloud Storage",
                    "billing_cycle": "monthly",
                    "customer_since": "2022-03-15",
                },
                "technical": {
                    "app_version": "3.2.1",
                    "error_code": 503,
                    "error_message": "Sync Failed — connection timeout after 30s",
                    "affected_endpoint": "api.cloudstore.example.com",
                    "known_issue": True,
                    "workaround": "Use web interface at app.cloudstore.example.com",
                    "fix_eta": "v3.2.2 patch expected 2025-06-15",
                },
                "account_tier": "premium",
                "loyalty_years": 3,
            },
        )
        return Observation(
            ticket=ticket,
            instruction=(
                "This is a complex multi-issue ticket. The customer has TWO problems:\n"
                "1) A billing issue — she was charged twice ($49.99 x 2) for her "
                "subscription. One charge needs to be refunded.\n"
                "2) A technical issue — her desktop app shows Error 503 and cannot "
                "sync files. A workaround (web interface) and fix ETA (June 15, v3.2.2) "
                "are available in the metadata.\n\n"
                "You must address BOTH issues in your response. Acknowledge the "
                "frustration, resolve the billing issue (process refund), and provide "
                "the technical workaround plus ETA. Be thorough, empathetic, and "
                "professional."
            ),
            available_actions=list(ActionType),
            turn=0,
            max_turns=self.max_turns,
        )

    def expected_actions(self) -> List[ActionType]:
        return [ActionType.REFUND, ActionType.RESPOND, ActionType.RESOLVE]

    def expected_keywords(self) -> List[str]:
        return [
            "refund", "$49.99", "double charge",
            "sync", "Error 503", "web interface",
            "June 15", "v3.2.2", "apologize", "sorry",
            "loyal", "appreciate",
        ]

    def forbidden_phrases(self) -> List[str]:
        return [
            "only one issue",
            "not responsible",
            "contact your bank",
            "we can't help",
            "that's a known issue and we don't care",
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    EasyTask.task_id: EasyTask(),
    MediumTask.task_id: MediumTask(),
    HardTask.task_id: HardTask(),
}


def get_task(task_id: str) -> Task:
    """Retrieve a task by ID."""
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> List[Dict[str, str]]:
    """Return metadata for all available tasks."""
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
        }
        for t in TASKS.values()
    ]
