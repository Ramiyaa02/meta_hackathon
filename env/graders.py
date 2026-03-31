"""
Deterministic grading functions for the AI Customer Support Ticket System.

Each grader evaluates the agent's action on three axes:
  1. Correctness  — did the agent take the right type of action?
  2. Politeness   — is the response professional, empathetic, and appropriate?
  3. Completeness — does the response address all required aspects of the ticket?

Scores are continuous in [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import List

from env.models import Action, Observation, Reward, RewardBreakdown, ActionType
from env.tasks import Task


# ---------------------------------------------------------------------------
# Individual scoring helpers
# ---------------------------------------------------------------------------

def _score_correctness(
    action: Action,
    task: Task,
    observation: Observation,
) -> tuple[float, str]:
    """
    Score the correctness of the action type.

    Checks whether the agent chose an appropriate action type given the
    task's expected action sequence and the current turn.
    """
    expected = task.expected_actions()
    feedback_parts: List[str] = []

    # Base score from action type matching
    if action.action_type in expected:
        correctness = 0.7
        feedback_parts.append(f"Action type '{action.action_type.value}' is appropriate.")
    else:
        # Partial credit: RESPOND is almost always acceptable
        if action.action_type == ActionType.RESPOND:
            correctness = 0.4
            feedback_parts.append(
                f"A '{action.action_type.value}' action is acceptable but not optimal. "
                f"Preferred actions: {[a.value for a in expected]}."
            )
        else:
            correctness = 0.1
            feedback_parts.append(
                f"Action type '{action.action_type.value}' is not expected. "
                f"Expected one of: {[a.value for a in expected]}."
            )

    # Bonus for using REFUND when the task involves a refund
    keywords_lower = " ".join(task.expected_keywords()).lower()
    if "refund" in keywords_lower and action.action_type == ActionType.REFUND:
        correctness = min(1.0, correctness + 0.2)
        feedback_parts.append("Good — refund action matches the billing context.")

    # Bonus for using RESOLVE at the right time (last turn or after main actions)
    if action.action_type == ActionType.RESOLVE:
        correctness = min(1.0, correctness + 0.1)
        feedback_parts.append("Resolving the ticket is a strong closing action.")

    # Penalty for ESCALATE when it's not necessary
    if action.action_type == ActionType.ESCALATE:
        metadata = observation.ticket.metadata
        if metadata.get("return_eligible") or metadata.get("known_issue"):
            correctness = max(0.0, correctness - 0.2)
            feedback_parts.append(
                "Escalation may not be necessary — the issue appears resolvable at this level."
            )

    return correctness, " ".join(feedback_parts)


def _score_politeness(action: Action) -> tuple[float, str]:
    """
    Score the politeness / professionalism of the response text.

    Uses heuristic keyword analysis for deterministic scoring.
    """
    text = action.response.lower()
    feedback_parts: List[str] = []

    # --- Positive signals ---
    polite_markers = [
        ("apolog", 0.12, "Apology detected — good de-escalation."),
        ("sorry", 0.12, "'Sorry' shows empathy."),
        ("understand", 0.08, "Acknowledging understanding is professional."),
        ("appreciate", 0.08, "Expressing appreciation builds rapport."),
        ("thank", 0.06, "Gratitude is a positive touch."),
        ("happy to help", 0.06, "Offering help proactively is excellent."),
        ("dear", 0.04, "Polite greeting detected."),
        ("please", 0.04, "Polite language detected."),
        ("welcome", 0.03, "Welcoming tone detected."),
    ]

    politeness = 0.4  # Base score — just being professional is ~0.4

    for keyword, bonus, msg in polite_markers:
        if keyword in text:
            politeness = min(1.0, politeness + bonus)
            feedback_parts.append(msg)

    # --- Negative signals ---
    rude_markers = [
        (r"\bcalm down\b", -0.2, "Telling a customer to 'calm down' is condescending."),
        (r"\bnot my (problem|job|fault)\b", -0.2, "Deflecting responsibility is unprofessional."),
        (r"\bwhatever\b", -0.15, "'Whatever' is dismissive."),
        (r"\bidiot\b", -0.3, "Insulting the customer is unacceptable."),
        (r"\bstupid\b", -0.25, "Derogatory language detected."),
        (r"\bshut up\b", -0.3, "Telling customer to be quiet is unacceptable."),
        (r"\bdon'?t care\b", -0.2, "Expressing indifference is unprofessional."),
        (r"\byour fault\b", -0.15, "Blaming the customer is inappropriate."),
    ]

    for pattern, penalty, msg in rude_markers:
        if re.search(pattern, text):
            politeness = max(0.0, politeness + penalty)
            feedback_parts.append(msg)

    if not feedback_parts:
        feedback_parts.append("Response tone is neutral-professional.")

    return politeness, " ".join(feedback_parts)


def _score_completeness(
    action: Action,
    task: Task,
) -> tuple[float, str]:
    """
    Score completeness — how many expected keywords / topics are covered.

    Compares the response against the task's expected keyword list.
    """
    text = action.response.lower()
    expected_keywords = task.expected_keywords()
    feedback_parts: List[str] = []

    if not expected_keywords:
        return 0.8, "No specific keywords expected for this task."

    matched = 0
    missed: List[str] = []

    for kw in expected_keywords:
        if kw.lower() in text:
            matched += 1
        else:
            missed.append(kw)

    ratio = matched / len(expected_keywords) if expected_keywords else 0.0
    completeness = round(ratio, 2)

    feedback_parts.append(
        f"Matched {matched}/{len(expected_keywords)} expected keywords."
    )
    if missed:
        feedback_parts.append(f"Missing keywords: {missed}")

    # Bonus for response length (too short = incomplete)
    word_count = len(action.response.split())
    if word_count < 15:
        completeness = max(0.0, completeness - 0.2)
        feedback_parts.append(
            f"Response is very short ({word_count} words). Consider providing more detail."
        )
    elif word_count > 50:
        completeness = min(1.0, completeness + 0.1)
        feedback_parts.append("Response is thorough and detailed.")

    return completeness, " ".join(feedback_parts)


def _compute_penalty(action: Action, task: Task) -> tuple[float, str]:
    """
    Compute penalties for bad actions.

    Returns (penalty_value, feedback). Penalty is subtracted from the score.
    """
    penalty = 0.0
    feedback_parts: List[str] = []
    text = action.response.lower()

    # Check forbidden phrases
    for phrase in task.forbidden_phrases():
        if phrase.lower() in text:
            penalty += 0.15
            feedback_parts.append(
                f"Forbidden phrase detected: '{phrase}'. This harms customer trust."
            )

    # Penalize empty responses
    if len(action.response.strip()) == 0:
        penalty += 0.3
        feedback_parts.append("Empty response — the customer received no communication.")

    # Penalize resolving without responding when there's an angry customer
    if action.action_type == ActionType.RESOLVE and len(action.response.strip()) < 10:
        penalty += 0.15
        feedback_parts.append(
            "Resolving without a meaningful response is poor customer service."
        )

    return min(penalty, 1.0), " ".join(feedback_parts) if feedback_parts else "No penalties."


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def grade(action: Action, observation: Observation, task: Task) -> Reward:
    """
    Grade an agent's action on the given task.

    Returns a Reward with a continuous score in [0.0, 1.0] and a
    detailed breakdown of correctness, politeness, completeness, and
    any penalties applied.
    """
    correctness, corr_fb = _score_correctness(action, task, observation)
    politeness, polite_fb = _score_politeness(action)
    completeness, comp_fb = _score_completeness(action, task)
    penalty, penalty_fb = _compute_penalty(action, task)

    # Weighted combination
    raw_score = (
        0.35 * correctness
        + 0.30 * politeness
        + 0.35 * completeness
    )

    # Apply penalty
    final_score = max(0.0, min(1.0, raw_score - penalty))
    final_score = round(final_score, 4)

    # Build feedback summary
    feedback_lines = [
        f"[Correctness]  {correctness:.2f} — {corr_fb}",
        f"[Politeness]   {politeness:.2f} — {polite_fb}",
        f"[Completeness] {completeness:.2f} — {comp_fb}",
        f"[Penalty]     -{penalty:.2f} — {penalty_fb}",
        f"[Final Score]  {final_score:.4f}",
    ]

    return Reward(
        score=final_score,
        breakdown=RewardBreakdown(
            correctness=round(correctness, 4),
            politeness=round(politeness, 4),
            completeness=round(completeness, 4),
            penalty=round(penalty, 4),
        ),
        feedback="\n".join(feedback_lines),
    )
