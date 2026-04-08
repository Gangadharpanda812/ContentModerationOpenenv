# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Data models for the Content Moderation Openenv Environment.

# The content_moderation_openenv environment is a simple test environment that echoes back messages.
# """

# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# class ContentModerationOpenenvAction(Action):
#     """Action for the Content Moderation Openenv environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class ContentModerationOpenenvObservation(Observation):
#     """Observation from the Content Moderation Openenv environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")


"""
Core typed models for the Content Moderation OpenEnv environment.
All models follow the OpenEnv specification using Pydantic v2.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    APPROVE = "APPROVE"
    REMOVE = "REMOVE"
    AGE_RESTRICT = "AGE_RESTRICT"
    ADD_WARNING = "ADD_WARNING"
    ESCALATE = "ESCALATE"
    REQUEST_CONTEXT = "REQUEST_CONTEXT"


class ContentLabel(str, Enum):
    SAFE = "SAFE"
    VIOLATING = "VIOLATING"
    BORDERLINE = "BORDERLINE"
    SPAM = "SPAM"
    HARASSMENT = "HARASSMENT"
    HATE_SPEECH = "HATE_SPEECH"
    MISINFORMATION = "MISINFORMATION"
    SELF_HARM = "SELF_HARM"
    EXPLICIT = "EXPLICIT"
    EVASION = "EVASION"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Author / Context Models ──────────────────────────────────────────────────

class AuthorHistory(BaseModel):
    """Historical signal about the content author."""
    account_age_days: int = Field(..., description="Days since account creation")
    prior_violations: int = Field(0, description="Number of past policy violations")
    prior_appeals_won: int = Field(0, description="Successful appeals by author")
    follower_count: int = Field(0, description="Follower count (reach proxy)")
    verified: bool = Field(False, description="Whether account is verified")
    flagged_by_users: int = Field(0, description="Community flags on this post")


class PlatformContext(BaseModel):
    """Platform and distribution context for a piece of content."""
    surface: str = Field(..., description="Where content appears: feed/search/direct")
    audience_minors_possible: bool = Field(False, description="Could minors see this?")
    trending: bool = Field(False, description="Is content trending / amplified?")
    reported_count: int = Field(0, description="Number of user reports")
    country_restrictions: List[str] = Field(
        default_factory=list,
        description="ISO-3166 country codes with special rules"
    )


# ─── Core OpenEnv Models ──────────────────────────────────────────────────────

class ContentModerationOpenenvObservation(BaseModel):
    """
    OpenEnv Observation — returned by reset() and step().
    Contains everything the agent needs to make a moderation decision.
    """
    post_id: str = Field(..., description="Unique post identifier")
    content: str = Field(..., description="Raw text content of the post")
    author_history: AuthorHistory
    platform_context: PlatformContext
    queue_position: int = Field(..., description="1-indexed position in current queue")
    queue_size: int = Field(..., description="Total items in this episode's queue")
    step: int = Field(0, description="Current step number in the episode")
    score: float = Field(0.0, description="Running episode score")
    episode_done: bool = Field(False, description="Whether the episode is complete")
    task_id: str = Field(..., description="Active task identifier")
    hint: Optional[str] = Field(None, description="Optional hint for debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "post_id": "p_001",
                "content": "Check out this amazing weight-loss trick doctors hate!",
                "author_history": {"account_age_days": 3, "prior_violations": 2},
                "platform_context": {"surface": "feed", "trending": True},
                "queue_position": 1,
                "queue_size": 10,
                "step": 0,
                "cumulative_score": 0.0,
                "episode_done": False,
                "task_id": "basic_triage"
            }
        }


class ContentModerationOpenenvAction(BaseModel):
    """
    OpenEnv Action — what the agent submits via step().
    Combines enforcement action with classification label and reasoning.
    """
    action: ActionType = Field(..., description="Enforcement action to apply")
    label: ContentLabel = Field(..., description="Policy violation classification")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agent confidence score (0.0–1.0)"
    )
    reasoning: str = Field(
        "", max_length=500,
        description="Brief rationale for the decision (used in grading)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action": "REMOVE",
                "label": "SPAM",
                "confidence": 0.92,
                "reasoning": "Health misinformation with spam patterns and new account"
            }
        }


class Reward(BaseModel):
    """
    OpenEnv Reward — step-level reward signal with breakdown.
    Provides partial credit signals, not just binary win/lose.
    """
    value: float = Field(..., ge=-1.0, le=1.0, description="Step reward [-1, 1]")
    label_score: float = Field(0.0, ge=0.0, le=1.0, description="Label accuracy component")
    action_score: float = Field(0.0, ge=0.0, le=1.0, description="Action appropriateness")
    evasion_bonus: float = Field(0.0, ge=0.0, le=0.5, description="Bonus for catching evasion")
    confidence_penalty: float = Field(0.0, ge=-0.2, le=0.0, description="Overconfidence penalty")
    reasoning_bonus: float = Field(0.0, ge=0.0, le=0.1, description="Bonus for quality reasoning")
    explanation: str = Field("", description="Human-readable reward explanation")


class EpisodeState(BaseModel):
    """Full serialisable state of an ongoing episode (returned by state())."""
    task_id: str
    difficulty: TaskDifficulty
    step: int
    queue: List[Dict[str, Any]]
    current_index: int
    cumulative_reward: float
    rewards_history: List[float]
    actions_taken: List[Dict[str, Any]]
    done: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Return type from step() — bundles all OpenEnv outputs."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# Aliases for backward compatibility
Action = ContentModerationOpenenvAction
Observation = ContentModerationOpenenvObservation