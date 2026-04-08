# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Content Moderation Openenv Environment Implementation.

A real-world content moderation environment where AI agents must triage,
classify, and take action on user-generated content (UGC) across a simulated
social platform. Agents review posts, apply policy labels, escalate edge cases,
and manage a live moderation queue — mirroring workflows used by trust & safety
teams at scale.
"""

from uuid import uuid4
import random
import copy

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ContentModerationOpenenvAction, ContentModerationOpenenvObservation, AuthorHistory, PlatformContext
    from .graders import grade
    from .dataset import ALL_POSTS, TASK_TO_DIFFICULTY
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import ContentModerationOpenenvAction, ContentModerationOpenenvObservation, AuthorHistory, PlatformContext
    from server.graders import grade
    from server.dataset import ALL_POSTS, TASK_TO_DIFFICULTY


class ContentModerationOpenenvEnvironment(Environment):
    """
    Content Moderation OpenEnv Environment.

    Agents act as platform moderators, reviewing user-generated content and making
    enforcement decisions. Tasks range from basic triage to adversarial evasion detection.

    Example:
        >>> env = ContentModerationOpenenvEnvironment()
        >>> obs = env.reset()
        >>> print(obs.content)  # Post content to moderate
        >>> action = ContentModerationOpenenvAction(action="REMOVE", label="SPAM", confidence=0.9, reasoning="Spam content")
        >>> obs, reward, done, info = env.step(action)
    """

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = None):
        """Initialize the content moderation environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id or random.choice(list(TASK_TO_DIFFICULTY.keys()))
        self._difficulty = TASK_TO_DIFFICULTY[self._task_id]
        self._posts = []
        self._current_index = 0
        self._cumulative_score = 0.0
        self._rewards_history = []

    def reset(self) -> ContentModerationOpenenvObservation:
        """
        Reset the environment for a new episode.

        Returns:
            Initial observation with the first post to moderate.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = random.choice(list(TASK_TO_DIFFICULTY.keys()))
        self._difficulty = TASK_TO_DIFFICULTY[self._task_id]
        self._posts = copy.deepcopy(ALL_POSTS[self._difficulty])
        random.shuffle(self._posts)
        self._current_index = 0
        self._cumulative_score = 0.0
        self._rewards_history = []

        current_post = self._posts[self._current_index]
        return ContentModerationOpenenvObservation(
            post_id=current_post["id"],
            content=current_post["content"],
            author_history=AuthorHistory(
                account_age_days=random.randint(1, 365),
                prior_violations=random.randint(0, 5),
                follower_count=random.randint(0, 10000),
                verified=random.choice([True, False]),
                flagged_by_users=random.randint(0, 10)
            ),
            platform_context=PlatformContext(
                surface=random.choice(["feed", "search", "direct"]),
                audience_minors_possible=random.choice([True, False]),
                trending=random.choice([True, False]),
                reported_count=random.randint(0, 50)
            ),
            queue_position=1,
            queue_size=len(self._posts),
            step=0,
            score=0.0,
            episode_done=False,
            task_id=self._task_id
        )

    def step(self, action: ContentModerationOpenenvAction):  # type: ignore[override]
        """
        Execute a moderation action on the current post.

        Args:
            action: The agent's moderation decision.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self._state.step_count += 1

        current_post = self._posts[self._current_index]
        ground_truth = {
            "gt_label": current_post["ground_truth_label"],
            "gt_action": current_post["ground_truth_action"],
            "is_evasion": current_post.get("is_evasion", False)
        }
        reward = grade(self._task_id, action, ground_truth)

        self._cumulative_score += reward.value
        self._rewards_history.append(reward.value)

        self._current_index += 1
        done = self._current_index >= len(self._posts)

        if not done:
            next_post = self._posts[self._current_index]
            observation = ContentModerationOpenenvObservation(
                post_id=next_post["id"],
                content=next_post["content"],
                author_history=AuthorHistory(
                    account_age_days=random.randint(1, 365),
                    prior_violations=random.randint(0, 5),
                    follower_count=random.randint(0, 10000),
                    verified=random.choice([True, False]),
                    flagged_by_users=random.randint(0, 10)
                ),
                platform_context=PlatformContext(
                    surface=random.choice(["feed", "search", "direct"]),
                    audience_minors_possible=random.choice([True, False]),
                    trending=random.choice([True, False]),
                    reported_count=random.randint(0, 50)
                ),
                queue_position=self._current_index + 1,
                queue_size=len(self._posts),
                step=self._state.step_count,
                score=self._cumulative_score,
                episode_done=False,
                task_id=self._task_id
            )
        else:
            observation = ContentModerationOpenenvObservation(
                post_id="",
                content="",
                author_history=AuthorHistory(account_age_days=0, prior_violations=0, follower_count=0, verified=False, flagged_by_users=0),
                platform_context=PlatformContext(surface="feed", audience_minors_possible=False, trending=False, reported_count=0),
                queue_position=0,
                queue_size=0,
                step=self._state.step_count,
                score=self._cumulative_score,
                episode_done=True,
                task_id=self._task_id
            )

        return observation, reward, done, {}

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

"""
ContentModerationEnv — Full OpenEnv-compliant environment.

Implements:
  reset(task_id) -> Observation
  step(action)   -> StepResult(observation, reward, done, info)
  state()        -> EpisodeState
"""

import copy
import random
from typing import Any, Dict, List, Optional


class ContentModerationEnv:
    """
    A real-world content moderation environment.

    The agent acts as a platform trust & safety moderator reviewing a queue
    of user-generated content posts. For each post the agent must:
      1. Classify the content with a ContentLabel
      2. Choose an ActionType (enforcement decision)
      3. Provide a confidence score and brief reasoning

    Three tasks of increasing difficulty:
      - basic_triage          (easy)   : 10 clear-cut posts, label accuracy focus
      - policy_enforcement    (medium) : 10 nuanced posts, action selection matters
      - adversarial_moderation (hard)  : 10 posts with obfuscation/evasion

    Reward is provided at every step (non-sparse), with partial credit for
    approximately-correct decisions and penalties for dangerous mis-classifications.
    """

    VALID_TASKS = list(TASK_TO_DIFFICULTY.keys())

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._seed = seed

        # Episode state (populated by reset)
        self._task_id: str = ""
        self._difficulty: str = ""
        self._queue: List[Dict[str, Any]] = []
        self._step: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._rewards_history: List[float] = []
        self._actions_taken: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # OpenEnv Interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "basic_triage") -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'basic_triage', 'policy_enforcement', 'adversarial_moderation'

        Returns:
            Initial Observation (first post in the queue)
        """
        if task_id not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task_id '{task_id}'. Must be one of: {self.VALID_TASKS}"
            )

        self._task_id = task_id
        self._difficulty = TASK_TO_DIFFICULTY[task_id]

        # Load and (optionally) shuffle the post queue for this difficulty
        raw_posts = copy.deepcopy(ALL_POSTS[self._difficulty])
        if self._seed is None:
            self._rng.shuffle(raw_posts)

        self._queue = raw_posts
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._rewards_history = []
        self._actions_taken = []

        return self._build_observation(self._step)

    def step(self, action: Action) -> StepResult:
        """
        Apply an agent action to the current post.

        Args:
            action: Action pydantic model with action, label, confidence, reasoning

        Returns:
            StepResult containing (observation, reward, done, info)

        Raises:
            RuntimeError if called before reset() or after episode ends
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )
        if self._step >= len(self._queue):
            raise RuntimeError("Step index out of bounds. Episode should be done.")

        # Grade the current action against ground truth
        current_post = self._queue[self._step]
        reward = grade(self._task_id, action, current_post)

        # Record action
        self._rewards_history.append(reward.value)
        self._cumulative_reward += reward.value
        self._actions_taken.append({
            "step": self._step,
            "post_id": current_post["post_id"],
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "ground_truth": {
                "label": current_post["gt_label"],
                "action": current_post["gt_action"],
                "is_evasion": current_post.get("is_evasion", False),
            }
        })

        # Advance step
        self._step += 1
        self._done = self._step >= len(self._queue)

        # Build next observation (or terminal)
        next_obs = self._build_observation(self._step if not self._done else self._step - 1)
        next_obs.episode_done = self._done
        next_obs.cumulative_score = round(self._cumulative_reward, 4)

        info = {
            "step_reward": reward.value,
            "cumulative_reward": self._cumulative_reward,
            "reward_breakdown": reward.model_dump(),
            "post_id": current_post["post_id"],
            "ground_truth_label": current_post["gt_label"],
            "ground_truth_action": current_post["gt_action"],
            "is_evasion": current_post.get("is_evasion", False),
            "steps_remaining": max(0, len(self._queue) - self._step),
        }

        if self._done:
            info["episode_score"] = self._episode_score()
            info["summary"] = self._episode_summary()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EpisodeState:
        """
        Return the full, serialisable state of the current episode.
        Useful for checkpointing and debugging.
        """
        return EpisodeState(
            task_id=self._task_id,
            difficulty=TaskDifficulty(self._difficulty) if self._difficulty else TaskDifficulty.EASY,
            step=self._step,
            queue=[
                {k: v for k, v in p.items() if k not in ("author_history", "platform_context")}
                | {
                    "author_history": p["author_history"],
                    "platform_context": p["platform_context"],
                }
                for p in self._queue
            ],
            current_index=self._step,
            cumulative_reward=round(self._cumulative_reward, 4),
            rewards_history=self._rewards_history.copy(),
            actions_taken=self._actions_taken.copy(),
            done=self._done,
            metadata={
                "task_id": self._task_id,
                "queue_size": len(self._queue),
                "seed": self._seed,
            }
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(self, index: int) -> Observation:
        """Build an Observation from the post at queue[index]."""
        if index >= len(self._queue):
            index = len(self._queue) - 1

        post = self._queue[index]
        ah = post["author_history"]
        pc = post["platform_context"]

        return Observation(
            post_id=post["post_id"],
            content=post["content"],
            author_history=AuthorHistory(
                account_age_days=ah.get("account_age_days", 30),
                prior_violations=ah.get("prior_violations", 0),
                prior_appeals_won=ah.get("prior_appeals_won", 0),
                follower_count=ah.get("follower_count", 0),
                verified=ah.get("verified", False),
                flagged_by_users=ah.get("flagged_by_users", 0),
            ),
            platform_context=PlatformContext(
                surface=pc.get("surface", "feed"),
                audience_minors_possible=pc.get("audience_minors_possible", False),
                trending=pc.get("trending", False),
                reported_count=pc.get("reported_count", 0),
                country_restrictions=pc.get("country_restrictions", []),
            ),
            queue_position=index + 1,
            queue_size=len(self._queue),
            step=index,
            cumulative_score=round(self._cumulative_reward, 4),
            episode_done=self._done,
            task_id=self._task_id,
        )

    def _episode_score(self) -> float:
        """Normalise cumulative reward to [0, 1] for final episode score."""
        if not self._rewards_history:
            return 0.0
        # Rewards are in [-1, 1]; shift and normalise to [0, 1]
        raw = sum(self._rewards_history)
        max_possible = len(self._rewards_history) * 1.0
        min_possible = len(self._rewards_history) * -1.0
        normalised = (raw - min_possible) / (max_possible - min_possible)
        return round(normalised, 4)

    def _episode_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the completed episode."""
        if not self._actions_taken:
            return {}

        correct_labels = sum(
            1 for a in self._actions_taken
            if a["action"]["label"] == a["ground_truth"]["label"]
        )
        correct_actions = sum(
            1 for a in self._actions_taken
            if a["action"]["action"] == a["ground_truth"]["action"]
        )
        evasion_detected = sum(
            1 for a in self._actions_taken
            if a["ground_truth"]["is_evasion"]
            and a["action"]["label"] in ("EVASION", a["ground_truth"]["label"])
        )
        evasion_total = sum(
            1 for a in self._actions_taken
            if a["ground_truth"]["is_evasion"]
        )

        return {
            "task_id": self._task_id,
            "total_steps": len(self._actions_taken),
            "label_accuracy": round(correct_labels / len(self._actions_taken), 3),
            "action_accuracy": round(correct_actions / len(self._actions_taken), 3),
            "evasion_recall": (
                round(evasion_detected / evasion_total, 3) if evasion_total > 0 else None
            ),
            "mean_reward": round(sum(self._rewards_history) / len(self._rewards_history), 4),
            "episode_score_0_1": self._episode_score(),
        }

    def render(self) -> str:
        """Return a text representation of the current queue state."""
        lines = [
            f"=== ContentModerationEnv | task={self._task_id} | step={self._step}/{len(self._queue)} ===",
            f"Cumulative reward: {self._cumulative_reward:.3f}",
        ]
        if self._step < len(self._queue):
            post = self._queue[self._step]
            lines.append(f"Current post [{post['post_id']}]: {post['content'][:80]}...")
        if self._done:
            lines.append(f"[DONE] Episode score: {self._episode_score():.4f}")
        return "\n".join(lines)