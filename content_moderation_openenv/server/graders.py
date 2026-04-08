"""
Graders for Content Moderation OpenEnv.

Each grader is deterministic and returns a Reward with partial credit signals.
Graders follow the schema: grade(action, ground_truth, context) -> Reward
"""

from __future__ import annotations
from typing import Any, Dict
try:
    from ..models import Action, Reward, ActionType, ContentLabel
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import Action, Reward, ActionType, ContentLabel


# ─── Severity / compatibility maps ───────────────────────────────────────────

# Numerical severity for actions (used for partial credit)
ACTION_SEVERITY: Dict[str, float] = {
    ActionType.APPROVE: 0.0,
    ActionType.ADD_WARNING: 0.25,
    ActionType.AGE_RESTRICT: 0.5,
    ActionType.REQUEST_CONTEXT: 0.4,
    ActionType.ESCALATE: 0.75,
    ActionType.REMOVE: 1.0,
}

# Violation severity for labels (used for partial credit)
LABEL_SEVERITY: Dict[str, float] = {
    ContentLabel.SAFE: 0.0,
    ContentLabel.BORDERLINE: 0.3,
    ContentLabel.SPAM: 0.5,
    ContentLabel.MISINFORMATION: 0.6,
    ContentLabel.EVASION: 0.7,
    ContentLabel.HARASSMENT: 0.75,
    ContentLabel.EXPLICIT: 0.8,
    ContentLabel.SELF_HARM: 0.85,
    ContentLabel.HATE_SPEECH: 0.9,
    ContentLabel.VIOLATING: 0.8,
}

# Compatible label groups (partial credit for nearby labels)
LABEL_GROUPS = [
    {ContentLabel.SAFE},
    {ContentLabel.BORDERLINE},
    {ContentLabel.SPAM},
    {ContentLabel.MISINFORMATION, ContentLabel.EVASION},
    {ContentLabel.HARASSMENT},
    {ContentLabel.HATE_SPEECH, ContentLabel.HARASSMENT},
    {ContentLabel.SELF_HARM},
    {ContentLabel.EXPLICIT},
    {ContentLabel.VIOLATING, ContentLabel.HARASSMENT, ContentLabel.HATE_SPEECH,
     ContentLabel.SPAM, ContentLabel.MISINFORMATION, ContentLabel.SELF_HARM,
     ContentLabel.EXPLICIT, ContentLabel.EVASION},
]

# Compatible action pairs (partial credit)
ACTION_PAIRS: Dict[str, frozenset] = {
    ActionType.REMOVE: frozenset({ActionType.REMOVE, ActionType.ESCALATE}),
    ActionType.ESCALATE: frozenset({ActionType.ESCALATE, ActionType.REMOVE}),
    ActionType.ADD_WARNING: frozenset({ActionType.ADD_WARNING, ActionType.AGE_RESTRICT, ActionType.REQUEST_CONTEXT}),
    ActionType.AGE_RESTRICT: frozenset({ActionType.AGE_RESTRICT, ActionType.ADD_WARNING}),
    ActionType.REQUEST_CONTEXT: frozenset({ActionType.REQUEST_CONTEXT, ActionType.ADD_WARNING, ActionType.ESCALATE}),
    ActionType.APPROVE: frozenset({ActionType.APPROVE}),
}


def _label_score(predicted: str, ground_truth: str) -> float:
    """Compute label accuracy with partial credit for related labels."""
    if predicted == ground_truth:
        return 1.0
    # Partial credit if in same compatibility group
    for group in LABEL_GROUPS:
        if ground_truth in group and predicted in group:
            return 0.5
    # Partial credit for severity proximity
    pred_sev = LABEL_SEVERITY.get(predicted, 0.5)
    gt_sev = LABEL_SEVERITY.get(ground_truth, 0.5)
    proximity = 1.0 - abs(pred_sev - gt_sev)
    if proximity > 0.85:
        return 0.3
    # Penalise calling a clear violation SAFE
    if predicted == ContentLabel.SAFE and gt_sev > 0.5:
        return -0.2
    return 0.0


def _action_score(predicted: str, ground_truth: str, gt_label: str) -> float:
    """Compute action appropriateness with partial credit."""
    if predicted == ground_truth:
        return 1.0
    # Check if action is in compatible set for ground truth
    compat = ACTION_PAIRS.get(ground_truth, frozenset({ground_truth}))
    if predicted in compat:
        return 0.6
    # Directional severity error
    pred_sev = ACTION_SEVERITY.get(predicted, 0.5)
    gt_sev = ACTION_SEVERITY.get(ground_truth, 0.5)
    sev_diff = abs(pred_sev - gt_sev)
    # Penalise under-moderation of high-severity content more than over-moderation
    gt_label_sev = LABEL_SEVERITY.get(gt_label, 0.5)
    if pred_sev < gt_sev and gt_label_sev > 0.7:
        # Under-moderating dangerous content — harsh penalty
        return max(-0.3, 0.4 - sev_diff * 1.5)
    if sev_diff <= 0.25:
        return 0.4
    if sev_diff <= 0.5:
        return 0.2
    return 0.0


def _reasoning_bonus(reasoning: str, is_evasion: bool, gt_label: str) -> float:
    """Small bonus for reasoning that demonstrates policy understanding."""
    if not reasoning or len(reasoning) < 20:
        return 0.0
    score = 0.0
    reasoning_lower = reasoning.lower()
    # Bonus for evasion detection in reasoning
    if is_evasion and any(
        kw in reasoning_lower
        for kw in ["evasion", "dog-whistle", "coded", "obfuscat", "bypass", "evad", "disguised"]
    ):
        score += 0.05
    # Bonus for policy-grounded reasoning
    if any(kw in reasoning_lower for kw in ["policy", "violat", "guideline", "rule"]):
        score += 0.03
    # Bonus for context consideration
    if any(kw in reasoning_lower for kw in ["context", "history", "account", "reach", "trending"]):
        score += 0.02
    return min(score, 0.10)


def _confidence_penalty(confidence: float, label_correct: bool, action_correct: bool) -> float:
    """Penalise overconfidence on wrong answers."""
    if label_correct and action_correct:
        return 0.0  # No penalty for confident correct answers
    if confidence > 0.8 and not label_correct:
        return -0.1 * (confidence - 0.8) / 0.2
    if confidence > 0.9 and not action_correct:
        return -0.05
    return 0.0


# ─── Task Graders ─────────────────────────────────────────────────────────────

def grade_basic_triage(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 1 (Easy): Basic Triage Grader.
    Focuses primarily on label accuracy. Action matters but has less weight.
    """
    gt_label = ground_truth["gt_label"]
    gt_action = ground_truth["gt_action"]

    ls = _label_score(action.label, gt_label)
    as_ = _action_score(action.action, gt_action, gt_label)

    # For easy tier: label is 70% of score, action 30%
    base_value = ls * 0.70 + as_ * 0.30
    conf_pen = _confidence_penalty(action.confidence, ls >= 0.8, as_ >= 0.6)
    reason_bon = _reasoning_bonus(action.reasoning, ground_truth.get("is_evasion", False), gt_label)

    total = max(-1.0, min(1.0, base_value + conf_pen + reason_bon))

    return Reward(
        value=round(total, 4),
        label_score=max(0.0, ls),
        action_score=max(0.0, as_),
        evasion_bonus=0.0,
        confidence_penalty=conf_pen,
        reasoning_bonus=reason_bon,
        explanation=(
            f"Label: {'✓' if ls == 1.0 else '~' if ls > 0 else '✗'} "
            f"(predicted={action.label}, truth={gt_label}, score={ls:.2f}) | "
            f"Action: {'✓' if as_ == 1.0 else '~' if as_ > 0 else '✗'} "
            f"(predicted={action.action}, truth={gt_action}, score={as_:.2f})"
        )
    )


def grade_policy_enforcement(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 2 (Medium): Policy Enforcement Grader.
    Label and action are equally weighted. Penalises dangerous under-moderation.
    """
    gt_label = ground_truth["gt_label"]
    gt_action = ground_truth["gt_action"]

    ls = _label_score(action.label, gt_label)
    as_ = _action_score(action.action, gt_action, gt_label)

    # For medium tier: equal weight, plus context matters
    base_value = ls * 0.50 + as_ * 0.50
    conf_pen = _confidence_penalty(action.confidence, ls >= 0.8, as_ >= 0.6)
    reason_bon = _reasoning_bonus(action.reasoning, ground_truth.get("is_evasion", False), gt_label)

    # Extra penalty for approving things that should be removed
    if (action.action == ActionType.APPROVE
            and gt_action in (ActionType.REMOVE, ActionType.ESCALATE)
            and LABEL_SEVERITY.get(gt_label, 0) > 0.7):
        base_value -= 0.25

    total = max(-1.0, min(1.0, base_value + conf_pen + reason_bon))

    return Reward(
        value=round(total, 4),
        label_score=max(0.0, ls),
        action_score=max(0.0, as_),
        evasion_bonus=0.0,
        confidence_penalty=conf_pen,
        reasoning_bonus=reason_bon,
        explanation=(
            f"Label: {ls:.2f} (pred={action.label}, truth={gt_label}) | "
            f"Action: {as_:.2f} (pred={action.action}, truth={gt_action}) | "
            f"base={base_value:.3f}"
        )
    )


def grade_adversarial_moderation(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 3 (Hard): Adversarial & Evasion Detection Grader.
    All of label, action, and evasion detection matter. Heavily penalises
    missing evasion attempts (false negatives on adversarial content).
    """
    gt_label = ground_truth["gt_label"]
    gt_action = ground_truth["gt_action"]
    is_evasion = ground_truth.get("is_evasion", False)

    ls = _label_score(action.label, gt_label)
    as_ = _action_score(action.action, gt_action, gt_label)

    # Evasion detection bonus
    evasion_bonus = 0.0
    evasion_missed_penalty = 0.0
    if is_evasion:
        if action.label == ContentLabel.EVASION:
            evasion_bonus = 0.20  # Explicit evasion detection rewarded
        elif action.label == gt_label:
            evasion_bonus = 0.10  # Got the right label even if not marked as evasion
        elif action.label == ContentLabel.SAFE:
            evasion_missed_penalty = -0.30  # Calling evasion content SAFE is very bad

    base_value = ls * 0.40 + as_ * 0.40 + evasion_bonus + evasion_missed_penalty
    conf_pen = _confidence_penalty(action.confidence, ls >= 0.8, as_ >= 0.6)
    reason_bon = _reasoning_bonus(action.reasoning, is_evasion, gt_label)

    # Penalise over-moderation (approving safe OR removing safe content)
    if gt_label == ContentLabel.SAFE and action.action == ActionType.REMOVE:
        base_value -= 0.20

    total = max(-1.0, min(1.0, base_value + conf_pen + reason_bon))

    return Reward(
        value=round(total, 4),
        label_score=max(0.0, ls),
        action_score=max(0.0, as_),
        evasion_bonus=evasion_bonus,
        confidence_penalty=conf_pen,
        reasoning_bonus=reason_bon,
        explanation=(
            f"Label: {ls:.2f} (pred={action.label}, truth={gt_label}) | "
            f"Action: {as_:.2f} (pred={action.action}, truth={gt_action}) | "
            f"Evasion: {'detected' if evasion_bonus > 0 else ('missed!' if is_evasion else 'n/a')} "
            f"(bonus={evasion_bonus:.2f}) | total={total:.3f}"
        )
    )


GRADERS = {
    "basic_triage": grade_basic_triage,
    "policy_enforcement": grade_policy_enforcement,
    "adversarial_moderation": grade_adversarial_moderation,
}


def grade(task_id: str, action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """Dispatch to the correct grader for a task."""
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(GRADERS.keys())}")
    return grader(action, ground_truth)