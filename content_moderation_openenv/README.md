---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - content-moderation
  - trust-and-safety
  - reinforcement-learning
  - agent-evaluation
---

# 🛡️ Content Moderation OpenEnv

A **real-world content moderation environment** for training and evaluating AI agents on trust & safety tasks. Agents act as platform moderators, reviewing user-generated content and making enforcement decisions — mirroring workflows used by T&S teams at scale.

---

## 🎯 Why This Environment?

Content moderation is one of the most consequential AI tasks:
- Over **500 hours** of video are uploaded to YouTube every minute
- Facebook's T&S team reviews **millions** of posts daily
- Mistakes cost real human welfare (missed self-harm content) or free expression (false positives)

This environment gives RL researchers a **principled, graded, multi-signal benchmark** for agent evaluation in this domain — a gap in current OpenEnv offerings.

---

## 📐 Environment Design

### Observation Space

```python
class Observation(BaseModel):
    post_id: str              # Unique post identifier
    content: str              # Raw text of the post
    author_history: AuthorHistory
    #   - account_age_days: int
    #   - prior_violations: int
    #   - follower_count: int
    #   - verified: bool
    #   - flagged_by_users: int
    platform_context: PlatformContext
    #   - surface: str         (feed/search/direct)
    #   - audience_minors_possible: bool
    #   - trending: bool
    #   - reported_count: int
    queue_position: int        # 1-indexed position in queue
    queue_size: int            # Total items this episode
    step: int
    cumulative_score: float
    episode_done: bool
    task_id: str
```

### Action Space

```python
class Action(BaseModel):
    action: ActionType         # APPROVE | REMOVE | AGE_RESTRICT | ADD_WARNING | ESCALATE | REQUEST_CONTEXT
    label: ContentLabel        # SAFE | VIOLATING | BORDERLINE | SPAM | HARASSMENT | HATE_SPEECH
                               # MISINFORMATION | SELF_HARM | EXPLICIT | EVASION
    confidence: float          # 0.0 – 1.0
    reasoning: str             # Brief explanation (improves reward)
```

### Reward Structure

Rewards are **non-sparse** and provided at every step:

| Component | Range | Description |
|-----------|-------|-------------|
| `label_score` | [-0.2, 1.0] | Exact match = 1.0; related category = 0.5; opposite = penalty |
| `action_score` | [-0.3, 1.0] | Correct action = 1.0; compatible action = 0.6 |
| `evasion_bonus` | [0.0, 0.20] | Bonus for detecting obfuscated/coded content |
| `confidence_penalty` | [-0.1, 0.0] | Penalises overconfident wrong answers |
| `reasoning_bonus` | [0.0, 0.10] | Bonus for policy-grounded reasoning |

**Final episode score is normalised to [0.0, 1.0].**

---

## 📋 Tasks

### Task 1: Basic Content Triage (Easy)
**Objective**: Classify 10 posts into correct violation categories.  
**Focus**: Label accuracy (70%) + action appropriateness (30%).  
**Examples**: Clear spam, explicit hate speech, safe everyday posts.  
**Baseline (gpt-4o-mini)**: **0.845**

### Task 2: Policy Enforcement with Actions (Medium)
**Objective**: Choose the correct enforcement action for nuanced cases.  
**Focus**: Equal label + action weighting. Edge cases where context determines severity.  
**Examples**: Disaster charity links (scam or genuine?), unverified medical advice, political speech.  
**Baseline (gpt-4o-mini)**: **0.713**

### Task 3: Adversarial & Evasion Detection (Hard)
**Objective**: Detect obfuscation, dog-whistles, and coded language.  
**Focus**: Evasion detection bonus, harsh penalty for missing adversarial content.  
**Examples**: Leet-speak substitutions, "just asking questions" framing, reversed slurs, hypothetical disclaimers.  
**Baseline (gpt-4o-mini)**: **0.606**

---

## 📊 Baseline Scores (gpt-4o-mini, seed=42)

| Task | Score (0–1) | Label Acc | Action Acc |
|------|-------------|-----------|------------|
| basic_triage | **0.845** | ~85% | ~80% |
| policy_enforcement | **0.713** | ~70% | ~65% |
| adversarial_moderation | **0.606** | ~60% | ~58% |
| **Overall** | **0.721** | | |

Difficulty progression confirmed: harder tasks score lower even for frontier models.

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/content-moderation-env
cd content-moderation-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860` for the interactive web UI.

### Docker

```bash
docker build -t content-moderation-env .
docker run -p 7860:7860 content-moderation-env
```

### Running the Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"task_id": "basic_triage", "seed": 42}` |
| `/step` | POST | Submit action. Body: `{"action": "REMOVE", "label": "SPAM", "confidence": 0.9, "reasoning": "..."}` |
| `/state` | GET | Full serialisable episode state |
| `/tasks` | GET | List all available tasks |
| `/action_space` | GET | Valid actions and labels |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

### Python Usage Example

```python
from env import ContentModerationEnv, Action, ActionType, ContentLabel

env = ContentModerationEnv(seed=42)
obs = env.reset("adversarial_moderation")

while not obs.episode_done:
    # Your agent logic here
    action = Action(
        action=ActionType.REMOVE,
        label=ContentLabel.EVASION,
        confidence=0.87,
        reasoning="Leet-speak substitution to evade hate speech filter"
    )
    result = env.step(action)
    print(f"Reward: {result.reward.value:.4f} — {result.reward.explanation}")
    obs = result.observation

print(f"Episode score: {result.info['episode_score']:.4f}")
```

---

## 📁 Project Structure

```
content-moderation-env/
├── openenv.yaml              # OpenEnv metadata spec
├── app.py                    # FastAPI server
├── requirements.txt
├── Dockerfile
├── env/
│   ├── __init__.py
│   ├── models.py             # Typed Pydantic models (Observation, Action, Reward)
│   ├── environment.py        # ContentModerationEnv (reset/step/state)
│   ├── dataset.py            # 30 posts with ground truth (10 per difficulty)
│   └── graders.py            # Deterministic graders with partial credit
├── baseline/
│   ├── run_baseline.py       # OpenAI-client baseline inference script
│   └── results.json          # Pre-computed reference scores
├── tests/
│   └── test_environment.py   # pytest test suite (30+ tests)
└── ui/
    └── index.html            # Interactive web demo
```

---

## 📜 License

MIT License. See LICENSE for details.