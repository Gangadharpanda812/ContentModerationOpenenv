#!/usr/bin/env python3
"""
Baseline inference script for Content Moderation OpenEnv.

Uses OpenAI API to run a language model agent against the environment.
Produces reproducible baseline scores for all tasks.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline_inference.py
"""

import os
import json
from typing import Dict, List, Any
from openai import OpenAI
from content_moderation_openenv.server.content_moderation_openenv_environment import ContentModerationOpenenvEnvironment
from content_moderation_openenv.models import ContentModerationOpenenvAction, ActionType, ContentLabel

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_action_from_llm(content: str, task_description: str) -> ContentModerationOpenenvAction:
    """
    Use LLM to decide on moderation action for a post.
    """
    prompt = f"""
You are a content moderation AI for a social platform. Your task is: {task_description}

Post to moderate: "{content}"

Based on platform policies, classify this content and choose an appropriate action.

Available labels: SAFE, VIOLATING, BORDERLINE, SPAM, HARASSMENT, HATE_SPEECH, MISINFORMATION, SELF_HARM, EXPLICIT, EVASION

Available actions: APPROVE, REMOVE, AGE_RESTRICT, ADD_WARNING, ESCALATE, REQUEST_CONTEXT

Respond with a JSON object containing:
- action: one of the available actions
- label: one of the available labels
- confidence: a number between 0.0 and 1.0
- reasoning: brief explanation

Example response:
{{
  "action": "REMOVE",
  "label": "SPAM",
  "confidence": 0.95,
  "reasoning": "This appears to be promotional spam"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a cost-effective model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=200
        )

        result = json.loads(response.choices[0].message.content.strip())
        return ContentModerationOpenenvAction(
            action=ActionType(result["action"]),
            label=ContentLabel(result["label"]),
            confidence=float(result["confidence"]),
            reasoning=result["reasoning"]
        )
    except Exception as e:
        print(f"LLM error: {e}")
        # Fallback to safe defaults
        return ContentModerationOpenenvAction(
            action=ActionType.APPROVE,
            label=ContentLabel.SAFE,
            confidence=0.5,
            reasoning="Fallback due to error"
        )

def run_baseline_evaluation() -> Dict[str, Any]:
    """
    Run baseline evaluation on all tasks.
    """
    results = {}

    task_descriptions = {
        "basic_triage": "Classify posts into SAFE/VIOLATING/BORDERLINE categories.",
        "policy_enforcement": "Choose appropriate enforcement actions for policy violations.",
        "adversarial_moderation": "Detect evasion attempts and subtle violations."
    }

    for task_id, description in task_descriptions.items():
        print(f"\nRunning baseline for task: {task_id}")

        env = ContentModerationOpenenvEnvironment(task_id=task_id)
        obs = env.reset()

        episode_rewards = []
        step_count = 0

        while not obs.episode_done:
            action = get_action_from_llm(obs.content, description)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward.value)
            step_count += 1

            if step_count > 50:  # Safety limit
                break

        total_score = sum(episode_rewards)
        avg_score = total_score / len(episode_rewards) if episode_rewards else 0

        results[task_id] = {
            "total_score": round(total_score, 3),
            "average_score": round(avg_score, 3),
            "steps": step_count,
            "rewards": episode_rewards
        }

        print(f"Task {task_id}: Score = {avg_score:.3f} (over {step_count} steps)")

    return results

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    print("Running Content Moderation OpenEnv baseline evaluation...")
    results = run_baseline_evaluation()

    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)

    for task, data in results.items():
        print(f"{task}: {data['average_score']:.3f} average score")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to baseline_results.json")