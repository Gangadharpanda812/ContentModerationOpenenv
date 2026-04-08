"""
Tests for Content Moderation OpenEnv.
Validates OpenEnv spec compliance, reward correctness, and episode management.

Run with: pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ContentModerationEnv, Action, ActionType, ContentLabel
from env.models import Observation, Reward, EpisodeState, StepResult
from env.graders import grade_basic_triage, grade_policy_enforcement, grade_adversarial_moderation


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return ContentModerationEnv(seed=42)


# ─── reset() Tests ────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("basic_triage")
        assert isinstance(obs, Observation)

    def test_reset_valid_tasks(self, env):
        for task_id in ["basic_triage", "policy_enforcement", "adversarial_moderation"]:
            obs = env.reset(task_id)
            assert obs.task_id == task_id

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Invalid task_id"):
            env.reset("invalid_task")

    def test_reset_queue_size(self, env):
        obs = env.reset("basic_triage")
        assert obs.queue_size == 10

    def test_reset_starts_at_step_zero(self, env):
        obs = env.reset("basic_triage")
        assert obs.step == 0
        assert obs.queue_position == 1

    def test_reset_cumulative_score_zero(self, env):
        obs = env.reset("basic_triage")
        assert obs.cumulative_score == 0.0

    def test_reset_not_done(self, env):
        obs = env.reset("basic_triage")
        assert obs.episode_done is False

    def test_double_reset_clean_state(self, env):
        env.reset("basic_triage")
        # Take a step
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.9)
        env.step(action)
        # Reset again
        obs = env.reset("policy_enforcement")
        assert obs.step == 0
        assert obs.cumulative_score == 0.0
        assert obs.task_id == "policy_enforcement"


# ─── step() Tests ─────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_step_result(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.8)
        result = env.step(action)
        assert isinstance(result, StepResult)

    def test_step_reward_in_range(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.REMOVE, label=ContentLabel.SPAM, confidence=0.9)
        result = env.step(action)
        assert -1.0 <= result.reward.value <= 1.0

    def test_step_advances_queue(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.8)
        result = env.step(action)
        assert result.observation.step == 1
        assert result.observation.queue_position == 2

    def test_step_before_reset_raises(self, env):
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.8)
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_episode_ends_after_queue_exhausted(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.7)
        result = None
        for _ in range(10):
            result = env.step(action)
        assert result.done is True
        assert result.observation.episode_done is True

    def test_step_after_done_raises(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.7)
        for _ in range(10):
            env.step(action)
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_info_contains_ground_truth(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.8)
        result = env.step(action)
        assert "ground_truth_label" in result.info
        assert "ground_truth_action" in result.info

    def test_final_step_has_episode_summary(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.7)
        result = None
        for _ in range(10):
            result = env.step(action)
        assert "episode_score" in result.info
        assert "summary" in result.info


# ─── state() Tests ────────────────────────────────────────────────────────────

class TestState:
    def test_state_returns_episode_state(self, env):
        env.reset("basic_triage")
        s = env.state()
        assert isinstance(s, EpisodeState)

    def test_state_reflects_current_step(self, env):
        env.reset("basic_triage")
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.8)
        env.step(action)
        env.step(action)
        s = env.state()
        assert s.step == 2

    def test_state_is_serialisable(self, env):
        import json
        env.reset("adversarial_moderation")
        s = env.state()
        # Should not raise
        json.dumps(s.model_dump())


# ─── Reward / Grader Tests ────────────────────────────────────────────────────

class TestGraders:
    SAFE_POST = {
        "post_id": "t_safe",
        "gt_label": "SAFE",
        "gt_action": "APPROVE",
        "is_evasion": False,
        "explanation": "Test"
    }
    SPAM_POST = {
        "post_id": "t_spam",
        "gt_label": "SPAM",
        "gt_action": "REMOVE",
        "is_evasion": False,
        "explanation": "Test"
    }
    EVASION_POST = {
        "post_id": "t_evasion",
        "gt_label": "EVASION",
        "gt_action": "REMOVE",
        "is_evasion": True,
        "explanation": "Test"
    }

    def test_perfect_label_and_action_scores_high(self):
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.9)
        reward = grade_basic_triage(action, self.SAFE_POST)
        assert reward.value >= 0.8

    def test_wrong_label_scores_lower(self):
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SPAM, confidence=0.9)
        reward = grade_basic_triage(action, self.SAFE_POST)
        assert reward.value < 0.5

    def test_calling_violation_safe_penalised(self):
        action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.95)
        reward = grade_policy_enforcement(action, self.SPAM_POST)
        assert reward.value < 0.0  # Should be penalised

    def test_reward_in_valid_range(self):
        for action_type in ActionType:
            for label in ContentLabel:
                action = Action(action=action_type, label=label, confidence=0.5)
                reward = grade_basic_triage(action, self.SAFE_POST)
                assert -1.0 <= reward.value <= 1.0, f"Reward out of range for {action_type}/{label}"

    def test_evasion_detection_bonus(self):
        action_with_evasion = Action(
            action=ActionType.REMOVE, label=ContentLabel.EVASION,
            confidence=0.8, reasoning="leet-speak evasion detected"
        )
        action_without = Action(
            action=ActionType.REMOVE, label=ContentLabel.HATE_SPEECH,
            confidence=0.8
        )
        reward_with = grade_adversarial_moderation(action_with_evasion, self.EVASION_POST)
        reward_without = grade_adversarial_moderation(action_without, self.EVASION_POST)
        assert reward_with.evasion_bonus > 0
        assert reward_with.value >= reward_without.value

    def test_reasoning_bonus_for_evasion_keywords(self):
        action = Action(
            action=ActionType.REMOVE, label=ContentLabel.EVASION,
            confidence=0.8,
            reasoning="This is evasion using coded language and dog-whistle policy bypass"
        )
        reward = grade_adversarial_moderation(action, self.EVASION_POST)
        assert reward.reasoning_bonus > 0

    def test_overconfident_wrong_answer_penalised(self):
        action_confident = Action(
            action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.99
        )
        action_uncertain = Action(
            action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.6
        )
        reward_confident = grade_basic_triage(action_confident, self.SPAM_POST)
        reward_uncertain = grade_basic_triage(action_uncertain, self.SPAM_POST)
        assert reward_confident.confidence_penalty < reward_uncertain.confidence_penalty


# ─── Model Validation Tests ───────────────────────────────────────────────────

class TestModels:
    def test_action_confidence_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=1.5)
        with pytest.raises(ValidationError):
            Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=-0.1)

    def test_action_invalid_enum(self):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            Action(action="INVALID_ACTION", label=ContentLabel.SAFE, confidence=0.5)

    def test_observation_serialisable(self, env):
        import json
        obs = env.reset("basic_triage")
        # Should not raise
        json.dumps(obs.model_dump())

    def test_reward_value_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Reward(value=1.5, label_score=0.5, action_score=0.5)


# ─── Full Episode Integration Tests ──────────────────────────────────────────

class TestFullEpisode:
    def test_full_easy_episode(self, env):
        obs = env.reset("basic_triage")
        total = 0.0
        for _ in range(10):
            action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.7)
            result = env.step(action)
            total += result.reward.value
        assert result.done is True
        assert "episode_score" in result.info

    def test_full_hard_episode_with_perfect_agent(self, env):
        """A perfect agent should score near 1.0 on all tasks."""
        from env.dataset import ALL_POSTS, TASK_TO_DIFFICULTY
        for task_id, difficulty in TASK_TO_DIFFICULTY.items():
            env2 = ContentModerationEnv(seed=42)
            env2.reset(task_id)
            posts = ALL_POSTS[difficulty]
            for post in posts:
                action = Action(
                    action=ActionType(post["gt_action"]),
                    label=ContentLabel(post["gt_label"]),
                    confidence=0.95,
                    reasoning="Perfect agent using ground truth"
                )
                result = env2.step(action)
            assert result.info["episode_score"] > 0.85, \
                f"Perfect agent should score >0.85 on {task_id}, got {result.info['episode_score']}"

    def test_difficulty_progression(self):
        """Harder tasks should be harder for a random agent."""
        scores = {}
        for task_id in ["basic_triage", "policy_enforcement", "adversarial_moderation"]:
            env = ContentModerationEnv(seed=42)
            env.reset(task_id)
            # Random-ish agent: always approves as safe
            for _ in range(10):
                action = Action(action=ActionType.APPROVE, label=ContentLabel.SAFE, confidence=0.5)
                result = env.step(action)
            scores[task_id] = result.info["episode_score"]

        # Basic triage should be easiest for a "safe" defaulting agent
        # (many safe posts in easy tier)
        assert scores["basic_triage"] >= scores["adversarial_moderation"], \
            f"Easy task should score >= hard task for naive agent. Got {scores}"