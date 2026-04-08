# Dataset definitions for Content Moderation OpenEnv

# Mapping of task IDs to their difficulties
TASK_TO_DIFFICULTY = {
    "basic_triage": "easy",
    "policy_enforcement": "medium",
    "adversarial_moderation": "hard"
}

# All posts organized by difficulty level
# Each post should have: id, content, ground_truth_label, ground_truth_action (for policy tasks), etc.
ALL_POSTS = {
    "easy": [
        {
            "id": "easy_1",
            "content": "This is a safe post about my day.",
            "ground_truth_label": "SAFE",
            "ground_truth_action": "APPROVE"
        },
        {
            "id": "easy_2",
            "content": "I love this community!",
            "ground_truth_label": "SAFE",
            "ground_truth_action": "APPROVE"
        },
        # Add more easy posts here
    ],
    "medium": [
        {
            "id": "medium_1",
            "content": "This post contains some inappropriate language.",
            "ground_truth_label": "VIOLATING",
            "ground_truth_action": "REMOVE"
        },
        # Add more medium posts here
    ],
    "hard": [
        {
            "id": "hard_1",
            "content": "Subtle evasion attempt with coded language.",
            "ground_truth_label": "EVASION",
            "ground_truth_action": "ESCALATE",
            "is_evasion": True
        },
        # Add more hard posts here
    ]
}